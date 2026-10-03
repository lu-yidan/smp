"""Report every registered alpha/settling run at one fixed checkpoint."""
import argparse, json
from pathlib import Path
import numpy as np
from analyze_a6_confirmatory_10k import metrics, sha
from queue_a6_clear_settle import ARMS, SEEDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--update',type=int,choices=(9500,10000,19999),default=19999)
    a=p.parse_args();rows=[];pending=[];initial=None;data={};hashes={}
    for seed in SEEDS:
        for arm in ARMS:
            for suite in ('nominal','upper130'):
                path=a.root/f'seed{seed}'/arm/'validation'/str(a.update)/suite
                if not (path/'summary.json').exists():
                    pending.append(dict(seed=seed,arm=arm,suite=suite));continue
                f=dict(np.load(path/'per_trial.npz'));init=dict(np.load(path/'initial.npz'))
                meta=json.loads((path/'launch.json').read_text())
                assert meta['num_envs']==2048 and meta['seed']==20261013
                if initial is None:initial=init
                assert all(np.array_equal(v,initial[k]) for k,v in init.items()),path
                for name in ('initial.npz','per_trial.npz','summary.json','launch.json'):
                    hashes[str((path/name).relative_to(a.root))]=sha(path/name)
                row=dict(seed=seed,arm=arm,suite=suite,checkpoint_sha256=meta['checkpoint_sha256'],
                         all_plates=metrics(f,f['scene']>0))
                for i,name in enumerate(('flat','guided','free')):row[name]=metrics(f,f['scene']==i)
                rows.append(row);data[seed,arm,suite]=f
    pairs=[]
    for seed in SEEDS:
        for alpha in ('a020','a050','a100'):
            for suite in ('nominal','upper130'):
                base=data.get((seed,'CS_'+alpha+'_base',suite));fixed=data.get((seed,'CS_'+alpha+'_settle',suite))
                if base is None or fixed is None:continue
                mask=base['scene']>0
                common=mask&(base['best_hold']>=10-1e-4)&(fixed['best_hold']>=10-1e-4)
                b=metrics(base,mask);f=metrics(fixed,mask)
                pairs.append(dict(seed=seed,alpha=alpha,suite=suite,
                    sr10_settle_minus_base=f['sr10']-b['sr10'],common_success_n=int(common.sum()),
                    common_success_base=metrics(base,common) if common.any() else None,
                    common_success_settle=metrics(fixed,common) if common.any() else None))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'analysis.json').write_text(json.dumps(dict(checkpoint_index=a.update,records=rows,
        paired=pairs,pending=pending,input_sha256=hashes,
        scope='All registered runs at a fixed checkpoint; no favorable-seed selection; reused validation bank'),indent=2)+'\n')
    print('Available suites:',len(rows),'Pending:',len(pending))

if __name__=='__main__':main()
