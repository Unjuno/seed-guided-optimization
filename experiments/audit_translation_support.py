"""Independent, post-hoc reconstruction of Issue89 calibration support."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
ROOT=Path(args.evidence_dir)/'downloaded'
records=[];checks=[]
for mp in sorted(ROOT.rglob('calibration_manifest.json')):
 m=json.loads(mp.read_text())
 for n,d in m['files'].items():assert hashlib.sha256((mp.parent/n).read_bytes()).hexdigest()==d
 for n,d in m['source_hashes'].items():assert hashlib.sha256((mp.parent/'source'/n).read_bytes()).hexdigest()==d
 df=pd.read_csv(next(mp.parent.glob('translation_subsets_*.csv.gz')),float_precision='round_trip')
 checks.append({'start':m['start'],'end':m['end'],'subsets':len(df),'runtime':m['runtime']})
 for (rep,step),v in df.groupby(['rep','step']):
  assert len(v)==84 and set(v.subset_id)==set(range(84))
  a=v.sort_values('subset_id');h=a.z_hard.to_numpy();p=a.z_param.to_numpy();t=a.translation_score.to_numpy();g=a.gradient_novelty.to_numpy()
  i,j=np.triu_indices(84,1);dh=abs(h[i]-h[j]);dp=abs(p[i]-p[j]);dt=abs(t[i]-t[j]);dg=abs(g[i]-g[j])
  ok=(dh<=.05+1e-12)&(dp<=.05+1e-12)
  r={'rep':int(rep),'step':int(step),'hp_pairs':int(ok.sum()),'min_possible_translation_gap':float(dt[ok].min()),'unrestricted_gradient_gap':float(dg[ok].max())}
  for c in [.05,.1,.2,.3,.5]:
   keep=ok&(dt<=c+1e-12);r[f'eligible_{c}']=int(keep.sum());r[f'max_g_{c}']=float(dg[keep].max()) if keep.any() else 0.
  records.append(r)
out=Path(args.output_dir);out.mkdir(exist_ok=True)
a=pd.DataFrame(records);a.to_csv(out/'support_posthoc.csv',index=False)
print(a.sort_values('min_possible_translation_gap',ascending=False).head(12).to_string(index=False))
print('threshold quantiles',a.min_possible_translation_gap.quantile([0,.5,.9,.95,.99,1]).to_dict())
for c in [.05,.1,.2,.3,.5]:
 col=f'max_g_{c}';print(c,'coverage',float((a[f'eligible_{c}']>0).mean()),'unconditional_gap',a[col].mean(),'conditional',a.loc[a[f'eligible_{c}']>0,col].mean(),'fallback',int((a[f'eligible_{c}']==0).sum()))
(out/'audit_metadata.json').write_text(json.dumps(checks,indent=2))
