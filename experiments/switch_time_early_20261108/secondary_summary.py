"""Secondary descriptive post-outcome accuracy/finite-pool tail summary (not tests)."""
from pathlib import Path
import csv,json,statistics
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
POL=('M','T40','T60','T80')
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
heldout=[];clean=[]
for p in range(12):
    heldout+=read(ROOT/f'runs/eval/pool{p}/heldout_counts.csv')
    clean+=read(ROOT/f'runs/eval/pool{p}/clean_counts.csv')
res={}
for q in POL:
    h=[r for r in heldout if r['policy']==q]
    c=[r for r in clean if r['policy']==q]
    assert len(h)==12*4*48 and len(c)==12*4
    perpool=[]
    for p in range(12):
        hh=[r for r in h if int(r['pool'])==p]
        cc=[r for r in c if int(r['pool'])==p]
        perpool.append(dict(pool=p,shift_accuracy=sum(int(r['correct']) for r in hh)/(len(hh)*809),clean_accuracy=sum(int(r['correct']) for r in cc)/(len(cc)*809)))
    res[q]=dict(shift_accuracy=sum(int(r['correct']) for r in h)/(len(h)*809),clean_accuracy=sum(int(r['correct']) for r in c)/(len(c)*809),perpool=perpool)
path=ROOT/'results/SECONDARY_ACCURACY.json'
if path.exists():raise RuntimeError('do not overwrite')
path.write_text(json.dumps(res,indent=2,sort_keys=True))
for q,d in res.items():print(q,d['shift_accuracy'],d['clean_accuracy'])
