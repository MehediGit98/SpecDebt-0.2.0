"""Reproduce specdebt using bundled source; no sibling archive needed."""
from pathlib import Path
import argparse
from pipeline import reproduce
from make_figures import main as figures
from compare_results import compare_results
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'reproduced');p.add_argument('--skip-comparison',action='store_true');a=p.parse_args()
    reproduce('specdebt',a.out)
    figures('specdebt',a.out,a.out/'figures')
    if not a.skip_comparison:
        n=compare_results('specdebt',ROOT/'results',a.out)
        print(f'PASS: {n} numeric fields match the archived results (rtol=1e-9, atol=1e-8).')
