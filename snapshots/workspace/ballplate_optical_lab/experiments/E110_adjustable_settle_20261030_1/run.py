"""Adjustable settling wait; frozen paired worker and independent RAW windows."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'E109_hold_effective_frame_20261030_1'
COMPUTE=ROOT.parent/'E107_paired_control_20261030_1'
sys.path.insert(1,str(COMPUTE))
import core as c
LAB=c.LAB
prior=c.load(COMPUTE/'run.py','e110_frozen_composition')
base_module=prior.base_module
def run_gui(cfg,destination,hardware,qa_hook=None):
    import page
    page.INITIAL_CONTROLS=dict(cfg.get('paired_ui',{}))
    return prior.run_gui(cfg,destination,hardware,qa_hook)


def main():
    p=argparse.ArgumentParser();p.add_argument('--hardware',action='store_true')
    p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--config',type=Path)
    p.add_argument('--config-sha256');a=p.parse_args()
    if not a.hardware:print('{"hardware_actions":0}');return
    if not a.light_blocked_confirmed or a.config.resolve()!=ROOT/'CONFIG.json' or os.environ.get('QT_QPA_PLATFORM')!='xcb':
        raise ValueError('explicit shielded sealed startup required')
    cfg=c.read(c.checked(dict(path=str(a.config),sha256=a.config_sha256)))
    for path,sha in cfg['source_sha256'].items():
        if c.digest(path)!=sha:raise ValueError('source changed: '+path)
    for key in ('contract','restart','recovery','display','caches','display_profile','surrogate_preview','paired_control'):c.checked(cfg[key])
    proof=c.read(c.checked(cfg['lifecycle_release']))
    if cfg.get('offline_preparation_only') is not False or not all(proof.get(k) is True for k in ('completed','old_normal_exit_verified','cold_baseline_verified','physical_shielding_confirmed')):
        raise ValueError('normal exit and cold verification required')
    base=base_module();base.load('e110_no_owner',base.DISPLAY/'run.py').no_owner()
    import page
    if Path(page.__file__).resolve()!=ROOT/'page.py':raise ValueError('corrected page not selected')
    run_gui(cfg,ROOT,True)


if __name__=='__main__':main()
