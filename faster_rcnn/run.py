"""Isolated experiment runner. Test is inaccessible from training commands."""
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch, torchvision
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from .core import Config, BirdDataset, resolve_data, train, audit, save_json, build_model, collate, configure_data
from .evaluation import evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
CONDITIONS={'clean':None,'dark_mild':.7,'dark_strong':.4,'blur_mild':1,'blur_strong':2}

def config(name):
    c=Config()
    c.horizontal_flip_probability=0 if name=='A' else .5
    c.photometric_probability=.5 if name in 'CDE' else 0
    c.blur_probability=.3 if name in 'DE' else 0
    return c

class Corrupted(BirdDataset):
    condition='clean'
    def __getitem__(self,i):
        x,t=super().__getitem__(i)
        im=Image.fromarray((x.permute(1,2,0).numpy()*255).round().astype(np.uint8))
        if self.condition.startswith('dark'):
            im=Image.fromarray((np.asarray(im,dtype=np.float32)*CONDITIONS[self.condition]).clip(0,255).astype(np.uint8))
        elif self.condition.startswith('blur'):
            im=im.filter(ImageFilter.GaussianBlur(CONDITIONS[self.condition]))
        return to_tensor(im),t

def inspect():
    OUT.mkdir(exist_ok=True)
    root=resolve_data()
    report={'root':str(root),'version':3,'splits':{},'issues':[], 'environment':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.cuda.is_available(),'cpu':platform.processor(),'threads':torch.get_num_threads()}}
    dims=[]
    for split in ('train','val','test'):
        ds=BirdDataset(root,split)
        count=empty=0
        for i,path in enumerate(ds.paths):
            try:
                x,t=ds[i]; count+=len(t['boxes']); empty+=int(len(t['boxes'])==0)
                if split=='train':
                    h,w=x.shape[-2:]; scale=min(800/min(h,w),1333/max(h,w))
                    wh=(t['boxes'][:,2:]-t['boxes'][:,:2]).numpy()*scale
                    dims.extend(wh.tolist())
            except Exception as e: report['issues'].append({'split':split,'image':str(path),'error':str(e)})
        labels=set(x.stem for x in (root/'labels'/split).glob('*.txt'))
        orphan=sorted(labels-set(x.stem for x in ds.paths))
        report['splits'][split]={'images':len(ds),'boxes':count,'classes':1 if count else 0,'negative_images':empty,'orphan_labels':orphan}
    wh=np.array(dims); area=np.sqrt(wh.prod(1)); ratio=wh[:,1]/wh[:,0]
    report['train_scaled_boxes']={k:np.percentile(v,[0,5,25,50,75,95,100]).tolist() for k,v in {'width':wh[:,0],'height':wh[:,1],'area':wh.prod(1),'sqrt_area':area,'h_over_w':ratio}.items()}
    report['anchor_size_coverage_4_to_181_fraction']=float(((area>=4)&(area<=181)).mean())
    report['test_access']='structural audit only; no predictions, selection or tuning'
    save_json(OUT/'audit.json',report)
    print(json.dumps(report,indent=2))
    if report['issues']: raise RuntimeError('Audit issues require explicit resolution')

def load(name):
    ck=torch.load(OUT/name/'best.pth',map_location='cpu',weights_only=True)
    c=Config(**ck['config']); model=build_model(c,pretrained=False)
    # Pretrained R50 FPN uses FrozenBatchNorm; retain it when restoring.
    from torchvision.ops import FrozenBatchNorm2d
    def restore(m):
        for k,v in list(m.named_children()):
            if isinstance(v,torch.nn.BatchNorm2d): setattr(m,k,FrozenBatchNorm2d(v.num_features))
            else: restore(v)
    restore(model); model.load_state_dict(ck['model']); model.eval()
    return model,c

def robust(name,split='val'):
    if split=='test':
        freeze=json.loads((OUT/'freeze.json').read_text())
        if name not in freeze['models']: raise ValueError('Model not frozen')
        digest=hashlib.sha256((OUT/name/'best.pth').read_bytes()).hexdigest()
        if digest!=freeze['models'][name]: raise ValueError('Frozen weights changed')
    dest=OUT/name/(split+'_robustness'); dest.mkdir(exist_ok=False)
    model,c=load(name); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    ds=Corrupted(resolve_data(),split); rows=[]
    for condition in CONDITIONS:
        ds.condition=condition
        metrics,pred=evaluate(model,DataLoader(ds,batch_size=1,collate_fn=collate),device,c)
        rows.append(dict(experiment=name,split=split,condition=condition,**metrics))
        save_json(dest/(condition+'_predictions.json'),{'coordinate_system':'original pixels xyxy','bird_label':1,'images':[p.relative_to(ds.root).as_posix() for p in ds.paths],'predictions':pred})
        save_json(dest/'metrics.json',rows)
    with (dest/'metrics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)

def timing(name):
    dest=OUT/name/'cpu_timing.json'
    if dest.exists(): raise FileExistsError(dest)
    protocol=json.loads((ROOT/'timing_protocol.json').read_text()); torch.set_num_threads(4)
    model,c=load(name); root=resolve_data(); files={p.stem:p for p in (root/'images/test').iterdir()}
    images=[Image.open(files[Path(n).stem]).convert('RGB') for n in protocol['images']]
    rows=[]
    with torch.inference_mode():
        for repeat in protocol['execution_order']:
            for i in range(5): model([to_tensor(images[i])])
            start=time.perf_counter()
            for i in repeat['image_indices']: model([to_tensor(images[i])])
            seconds=time.perf_counter()-start
            rows.append({'repeat':repeat['repeat'],'total_seconds':seconds,'seconds_per_image':seconds/30,'fps':30/seconds})
    save_json(dest,{'results':rows,'threads':4,'cpu':platform.processor(),'torch':torch.__version__,'images':protocol['images'],'scope':'PIL to tensor + normalization + resize + inference + NMS + original-coordinate mapping; excludes loading/decode'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','train','smoke','robust','freeze','test','timing']);parser.add_argument('--experiment',choices=list('ABCDE'),default='A');parser.add_argument('--data-root', help='Local version-3 dataset root');parser.add_argument('--data-config', help='JSON configuration, see data_config.example.json');a=parser.parse_args()
    configure_data(a.data_root, a.data_config)
    torch.set_num_threads(4)
    if a.command=='audit': inspect()
    elif a.command in ('train','smoke'):
        if a.experiment=='E': raise ValueError('Improvement must be specified from A-D validation evidence first')
        train(config(a.experiment),output=OUT/(a.experiment if a.command=='train' else 'smoke'),smoke=a.command=='smoke')
    elif a.command=='robust': robust(a.experiment)
    elif a.command=='freeze':
        # Must finish validation robustness before any final test access.
        models={}
        for n in 'ABCDE':
            if not (OUT/n/'val_robustness/metrics.json').exists(): raise ValueError('Complete A-D validation robustness first')
            models[n]=hashlib.sha256((OUT/n/'best.pth').read_bytes()).hexdigest()
        with (OUT/'freeze.json').open('x') as f: json.dump({'models':models,'selection':'validation mAP50-95; earliest tie','test_tuning':False},f,indent=2)
    elif a.command=='test': robust(a.experiment,'test')
    elif a.command=='timing': timing(a.experiment)

if __name__=='__main__': main()
