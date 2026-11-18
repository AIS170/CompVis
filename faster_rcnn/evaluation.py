import numpy as np
import torch
from .upstream import load_exact_evaluator, prediction_tp

@torch.inference_mode()
def evaluate(model, loader, device, config):
    ns, _, _ = load_exact_evaluator()
    model.eval()
    stats = {k: [] for k in ('tp','conf','pred_cls','target_cls')}
    records=[]
    tp=fp=ngt=0
    for images, targets in loader:
        transformed=[]
        hook=model.transform.register_forward_hook(lambda m,args,result: transformed.append((result[0].image_sizes,list(result[0].tensors.shape))))
        try: outputs=model([x.to(device) for x in images])
        finally: hook.remove()
        for image_index,(output,target) in enumerate(zip(outputs,targets)):
            output={k:v.cpu() for k,v in output.items()}
            matched=prediction_tp(ns,output['boxes'],output['labels'],target)
            stats['tp'].append(matched)
            stats['conf'].append(output['scores'].numpy())
            stats['pred_cls'].append(output['labels'].numpy())
            stats['target_cls'].append(target['labels'].numpy())
            # Re-match at the operating point with the same upstream matching rule.
            keep=output['scores']>=config.pr_confidence
            fixed=prediction_tp(ns,output['boxes'][keep],output['labels'][keep],target)
            ntp=int(fixed[:,0].sum()); tp+=ntp; fp+=int(keep.sum())-ntp
            ngt+=len(target['boxes'])
            records.append({'image_id':int(target['image_id']), 'input_hw':list(images[image_index].shape[-2:]), 'resized_hw':list(transformed[0][0][image_index]), 'padded_batch_shape':transformed[0][1],**{k:v.tolist() for k,v in output.items()}})
    args={k:np.concatenate(v,axis=0) for k,v in stats.items()}
    ap=ns['ap_per_class'](**args,plot=False,names={1:'bird'})[5]
    return dict(map_50=float(ap[:,0].mean()) if ap.size else 0.,map_50_95=float(ap.mean()) if ap.size else 0.,precision=tp/max(tp+fp,1),recall=tp/max(ngt,1),tp=tp,fp=fp,fn=ngt-tp),records
