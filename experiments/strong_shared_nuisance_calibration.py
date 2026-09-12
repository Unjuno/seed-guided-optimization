"""Issue116 Stage A: fresh strong-range calibration for shared nuisance."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import shared_nuisance_calibration as base

REPS=tuple(range(4200,4210)); TRAIN_SEEDS=tuple(range(103000,103064)); OFFSET=4610000000
SEVERITIES=(.90,1.20,1.60,2.00,2.50,3.00,4.00); N_TOL=.03; C_TOL=.10

def configure():
    base.REPS=REPS; base.TRAIN_SEEDS=TRAIN_SEEDS; base.OFFSET=OFFSET; base.SEVERITIES=SEVERITIES
    base.N_TOL=N_TOL; base.C_TOL=C_TOL
    base.PROTOCOL={"issue":116,"stage":"A","reps":REPS,"train_seeds":TRAIN_SEEDS,"offset":OFFSET,"stride":4099,
        "severities":SEVERITIES,"pattern_scale":.30,"K":base.K,"Q":base.Q,"batch":base.BATCH,"epochs":base.EPOCHS,
        "lr":base.LR,"wd":base.WD,"novelty_weight":base.NOV_W,"novelty_gain_tolerance":N_TOL,
        "candidate_loss_tolerance":C_TOL,"nuisance":"environment-shared deterministic zero-mean unit-SD random pixel pattern",
        "threads":1}
    base.PH=hashlib.sha256(json.dumps(base.PROTOCOL,sort_keys=True).encode()).hexdigest()
    base.SOURCES=("common.py","fixed_dose_response.py","shared_nuisance_calibration.py","strong_shared_nuisance_calibration.py")

def selftest():
    configure(); base.selftest(); assert base.SEVERITIES==SEVERITIES and base.REPS==REPS
    print('STRONG_SHARED_NUISANCE_SELFTEST_PASS',base.PH)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');a=p.parse_args();configure()
    if a.mode=='selftest':selftest()
    elif a.mode=='run':base.run(a.start,a.end,Path(a.output_dir))
    else:base.summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()
