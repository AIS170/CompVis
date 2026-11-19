"""Run A-D, validation-led single-change improvement, freeze, final tests."""
import csv, hashlib, json, time, traceback
from dataclasses import asdict
from .core import train, save_json, configure_data
from .run import OUT, config, robust, timing
from .analysis import failures


def status(stage,**kwargs):
    save_json(OUT/'status.json',dict(stage=stage,updated_unix=time.time(),**kwargs))
    print(stage,kwargs,flush=True)

def summary():
    rows=[]
    for name in 'ABCDE':
        if not (OUT/name/'summary.json').exists(): continue
        train_info=json.loads((OUT/name/'summary.json').read_text())
        cfg=json.loads((OUT/name/'config.json').read_text())['config']
        speed=json.loads((OUT/name/'cpu_timing.json').read_text()) if (OUT/name/'cpu_timing.json').exists() else None
        for split in ('val','test'):
            path=OUT/name/(split+'_robustness/metrics.json')
            if not path.exists(): continue
            for row in json.loads(path.read_text()):
                rows.append({**row,'selected_epoch':train_info['selected_epoch'],'training_seconds':train_info['training_seconds'],'cpu_seconds_per_image':sum(r['seconds_per_image'] for r in speed['results'])/3 if speed else '', 'cpu_fps':90/sum(r['total_seconds'] for r in speed['results']) if speed else '', 'config':json.dumps(cfg)})
    if rows:
        save_json(OUT/'results.json',rows)
        with (OUT/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    try:
        for name in 'ABCD':
            status('training',experiment=name)
            if not (OUT/name/'summary.json').exists(): train(config(name),output=OUT/name)
            status('validation_robustness',experiment=name)
            if not (OUT/name/'val_robustness/metrics.csv').exists(): robust(name)
            failures(name);summary()
        # Base selected solely by mean validation AP across fixed five conditions.
        records={n:json.loads((OUT/n/'val_robustness/metrics.json').read_text()) for n in 'ABCD'}
        base=max('ABCD',key=lambda n:sum(r['map_50_95'] for r in records[n])/5)
        by={r['condition']:r['map_50_95'] for r in records[base]}
        weakness='darkness' if by['dark_strong']<=by['blur_strong'] else 'blur'
        c=config(base)
        if weakness=='darkness': c.dark_probability=.3
        else: c.blur_probability=.6
        plan={'base':base,'weakness':weakness,'hypothesis':'Target the larger strong-corruption validation AP deficit with one training-only change','single_change':'dark_probability 0 -> .3, uniform multiplicative factor [.4,.7]' if weakness=='darkness' else f'blur_probability {config(base).blur_probability} -> .6','config':asdict(c),'selection_data':'validation only; test never consulted','initialization':'fresh COCO weights; same seed and epoch budget'}
        save_json(OUT/'improvement_plan.json',plan)
        status('training',experiment='E')
        if not (OUT/'E/summary.json').exists(): train(c,output=OUT/'E')
        if not (OUT/'E/val_robustness/metrics.csv').exists(): robust('E')
        failures('E');summary()
        e=json.loads((OUT/'E/val_robustness/metrics.json').read_text())
        base_by={r['condition']:r for r in records[base]}
        improvement_rows=[{'condition':r['condition'],'base':base,'base_ap50_95':base_by[r['condition']]['map_50_95'],'E_ap50_95':r['map_50_95'],'delta_ap50_95':r['map_50_95']-base_by[r['condition']]['map_50_95']} for r in e]
        save_json(OUT/'improvement_validation.json',improvement_rows)
        lines=['# Validation-only improvement analysis','',json.dumps(plan,indent=2),'','| Condition | Base AP50-95 | E AP50-95 | Delta |','|---|---:|---:|---:|']
        lines += [f"| {r['condition']} | {r['base_ap50_95']:.4f} | {r['E_ap50_95']:.4f} | {r['delta_ap50_95']:+.4f} |" for r in improvement_rows]
        lines += ['', 'These results describe one seed and do not establish statistical significance. Check validation failure figures before assigning causes. Test was not used in this analysis.']
        (OUT/'improvement_validation.md').write_text('\n'.join(lines),encoding='utf-8')
        candidates={**records,'E':e}
        selected=max('ABCDE',key=lambda n:sum(r['map_50_95'] for r in candidates[n])/5)
        freeze={'models':{n:hashlib.sha256((OUT/n/'best.pth').read_bytes()).hexdigest() for n in 'ABCDE'},'final_choice':selected,'selection':'maximum mean five-condition validation AP50-95; alphabetical tie','test_tuning':False,'improvement':plan}
        if not (OUT/'freeze.json').exists(): save_json(OUT/'freeze.json',freeze)
        for name in 'ABCDE':
            status('final_test',experiment=name)
            if not (OUT/name/'test_robustness/metrics.csv').exists(): robust(name,'test')
            status('cpu_timing',experiment=name)
            if not (OUT/name/'cpu_timing.json').exists(): timing(name)
            summary()
        status('complete',final_choice=selected)
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':
    import torch
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--data-config')
    args = parser.parse_args()
    configure_data(args.data_root, args.data_config)
    torch.set_num_threads(4)
    main()
