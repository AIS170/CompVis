"""Validation-only failure diagnostics and learning curves."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from .core import BirdDataset, resolve_data, save_json
from .run import OUT

def failures(name):
    dest=OUT/name/'failures'; dest.mkdir(exist_ok=True)
    ds=BirdDataset(resolve_data(),'val')
    for condition in ('clean','dark_strong','blur_strong'):
        data=json.loads((OUT/name/'val_robustness'/(condition+'_predictions.json')).read_text())
        candidates=[]
        for pred in data['predictions']:
            i=pred['image_id']; _,target=ds[i]; gt=target['boxes']
            boxes=torch.tensor(pred['boxes']).reshape(-1,4); scores=torch.tensor(pred['scores']); boxes=boxes[scores>=.25]; scores=scores[scores>=.25]
            used=set(); matched=[]
            for box in boxes:
                overlaps=box_iou(box[None],gt)[0] if len(gt) else torch.empty(0)
                options=[j for j in range(len(gt)) if j not in used and overlaps[j]>=.5]
                j=max(options,key=lambda j:float(overlaps[j])) if options else None
                matched.append(j)
                if j is not None: used.add(j)
            fn=len(gt)-len(used); fp=sum(j is None for j in matched)
            small=int((((gt[:,2:]-gt[:,:2]).prod(1))<32**2).sum())
            candidates.append(dict(index=i,fn=fn,fp=fp,gt=len(gt),small=small,matched=matched,used=list(used),boxes=boxes.tolist(),scores=scores.tolist()))
        selected=[]
        for key in ('fn','fp','gt','small'):
            for row in sorted(candidates,key=lambda x:-x[key])[:2]:
                if row['index'] not in [x['index'] for x in selected]: selected.append(row)
        for row in selected:
            i=row['index']; im=Image.open(ds.paths[i]).convert('RGB'); _,target=ds[i]
            if condition.startswith('dark'): im=Image.fromarray((np.asarray(im,dtype=np.float32)*.4).astype(np.uint8))
            if condition.startswith('blur'):
                from PIL import ImageFilter
                im=im.filter(ImageFilter.GaussianBlur(2))
            draw=ImageDraw.Draw(im)
            for j,box in enumerate(target['boxes'].tolist()): draw.rectangle(box,outline='lime' if j in row['used'] else 'orange',width=2)
            for box,score,match in zip(row['boxes'],row['scores'],row['matched']):
                draw.rectangle(box,outline='cyan' if match is not None else 'red',width=2);draw.text(box[:2],f'{score:.2f}',fill='white')
            im.save(dest/(condition+'_'+ds.paths[i].stem+'.png'))
        save_json(dest/(condition+'.json'),selected)
    (dest/'README.md').write_text('Ground truth: green=matched, orange=missed. Predictions: cyan=matched, red=false positive. Confidence >=0.25 and IoU >=0.50. Diagnostic score-greedy matching is for visualization only; reported metrics use upstream Ultralytics matching. Cases ranked using validation FN, FP, density and original-pixel small-box count. Crowding can cause suppression/overlap ambiguity; darkness reduces contrast; blur removes edges. These are hypotheses, not proven causes. Inspect images before quoting causes in the report.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((OUT/name/'history.json').read_text()); fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]);ax[0].set_title('Training loss')
    for key in ('map_50','map_50_95'): ax[1].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    ax[1].legend();fig.tight_layout();fig.savefig(dest/'learning_curves.png');plt.close(fig)
