import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import torch, torchvision
from PIL import Image
from .core import Config, build_model, horizontal_flip, save_json
from .upstream import load_exact_evaluator, prediction_tp, DEFAULT_WHEEL
from .run import OUT

def main():
    torch.set_num_threads(4)
    ns,files,functions=load_exact_evaluator()
    target={'boxes':torch.tensor([[1.,2.,7.,9.]]),'labels':torch.ones(1,dtype=torch.long)}
    image=Image.new('RGB',(20,10)); flipped,t=horizontal_flip(image,target)
    assert t['boxes'].tolist()==[[13.,2.,19.,9.]] and target['boxes'].tolist()==[[1.,2.,7.,9.]]
    tp=prediction_tp(ns,target['boxes'],target['labels'],target)
    assert tp.all() and tp.shape==(1,10)
    ap=ns['ap_per_class'](tp,np.array([.9]),np.array([1]),np.array([1]),plot=False,names={1:'bird'})[5]
    assert .99 < float(ap.mean()) <= 1
    m=build_model(Config(),pretrained=True)
    m.eval()
    with torch.no_grad(): transformed,_=m.transform([torch.zeros(3,300,600)],None)
    assert transformed.image_sizes==[(666,1333)]
    assert len(m.rpn.anchor_generator.sizes)==5 and m.roi_heads.detections_per_img==300
    OUT.mkdir(exist_ok=True)
    save_json(OUT/'verification.json',{'checks':'flip/box alignment, exact IoU matching, perfect-detection AP, actual resize, FPN levels, detection cap passed','example_resize_hw':transformed.image_sizes,'padded_shape':list(transformed.tensors.shape),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'cpu':platform.processor() or platform.machine(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,'amp':False,'gradient_accumulation':1,'threads':4,'wheel_sha256':hashlib.sha256(DEFAULT_WHEEL.read_bytes()).hexdigest(),'function_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in functions.items()}})
    print('Verification passed')
if __name__=='__main__': main()
