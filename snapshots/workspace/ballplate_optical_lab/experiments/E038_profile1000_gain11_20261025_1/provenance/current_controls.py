"""Explicit current-owner bootstrap/snapshot/zero; stdlib-only, no import IO.

No arm, SDK, parameter write, restart, file loading, or inherited qualification.
The historical preflight supplies window identities ONLY, not the new profile.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

LAB=Path('/home/user/Documents/YQH/工作/强化学习实验/mujoco_project/ballplate_optical_lab')
ROOT=LAB/'experiments/E038_profile1000_gain11_20261025_1'
SESSION=LAB/'experiments/E032_off_baseline_gui_20261024_1/GUI'
LEGACY=LAB.parent/'ballplate_fourframe96_deploy'
REFERENCE=LAB/'experiments/E036_mapping_revisit_20261025_1/B02/PREFLIGHT.json'
REFERENCE_SHA='efa53663846ebbab70097e283643e5f0262aed94b6d399ae72d413a30d087cb4'
NATIVE=LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py'
NATIVE_SHA='a5c3c7acf9618422f364a602653e02536bb12a0c34497f18455eb18e33d4330d'
PID=90503
START='1347071'
GAIN='10.881464004516602'


def _read(p):return json.loads(Path(p).read_text())
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _pin(p,d):
    p=Path(p)
    if not p.is_absolute() or p.is_symlink() or _sha(p)!=d:raise ValueError('explicit source/cache SHA differs: '+str(p))
    return p
def _write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def _heartbeat(h,active=None,settings=None):
    if (h.get('pid')!=PID or type(h.get('unix_ns'))is not int or not 0<=time.time_ns()-h['unix_ns']<5e9
            or h.get('hardware_fault') or h.get('live_camera')is not True or h.get('camera_reserved')is not True
            or any(h.get(k)is not False for k in('loop_active','adaptation_active','deployment_busy'))
            or h.get('spatial_calibration',{}).get('busy')is not False
            or type(h.get('projection_armed'))is not bool or not 0<=(h.get('latest_frame')or{}).get('age_seconds',-1)<1):
        raise RuntimeError('fresh healthy idle original camera owner required')
    if active is not None and h.get('camera_active_readback')!=active:raise RuntimeError('full current camera profile changed')
    if settings is not None and h.get('manual_settings')!=settings:raise RuntimeError('current manual settings changed')
    return h
def _caches():
    rows=[json.loads(x)for x in(SESSION/'events.jsonl').read_text().splitlines()];result={}
    for role,flag in(('dmd',False),('slm',True)):
        hits=[r for r in rows if r.get('state')=='manual_file_loaded'and r.get('SLM')is flag]
        if not hits:raise RuntimeError('explicit current file cache absent')
        result[role]=hits[-1];_pin(result[role]['path'],result[role]['sha256'])
    return result
def _guard(ui,c):
    from optical_lab import legacy_gui as old
    p=ui._current_profile;h=_heartbeat(ui.state(),p['active'],p['settings'])
    if old._start_ticks(PID)!=START or _caches()!=p['caches']:raise RuntimeError('owner lifetime/cache changed')
    for f,d in p['sources'].items():_pin(f,d)
    if c.read_roles()!=p['roles']:raise RuntimeError('physical display roles changed')
    for k in('actual_roi','optical_reference','geometry_frozen'):
        if h.get(k)!=p[k]:raise RuntimeError('current optical context changed: '+k)
    expected,cd,cs=c.expected_pixels(p['settings'],ui._dmd,ui._phase,h['projection_armed'],'file')
    if not all(ui._pixels.compare(tuple(ui.before['windows'][r]['rect']),expected[r])['matches']for r in('dmd','slm')):
        raise RuntimeError('current actual panel bytes mismatch')
    return h,cd,cs


def prepare_current(out,expected_active=None):
    """Explicit read-only X/AT-SPI bootstrap; return (ui, clients, native)."""
    import numpy as np
    from optical_lab import legacy_gui as old
    ref=_read(_pin(REFERENCE,REFERENCE_SHA));_pin(NATIVE,NATIVE_SHA)
    h=_heartbeat(_read(SESSION/'heartbeat.json'));runtime=_read(SESSION/'runtime_sources.json')
    baseline=dict(ref['heartbeat']['camera_active_readback'],ExposureTime='92.74482727050781',Gain=GAIN)
    active=baseline if expected_active is None else dict(expected_active)
    if set(active)!=set(baseline)or any(active[k]!=v for k,v in baseline.items()if k!='ExposureTime'):
        raise ValueError('only explicit exposure-only next-stage profile allowed')
    _heartbeat(h,active)
    s=h['manual_settings']
    if s['dmd']!='file'or s['slm']!='file'or s['x']!=-410 or s['y']!=4 or s['period']!=35 or s['full_panel']:
        raise RuntimeError('fixed current file/XY/period scope required')
    if old._start_ticks(PID)!=START or runtime.get('pid')!=PID:raise RuntimeError('original owner lifetime changed')
    sources={str(REFERENCE):REFERENCE_SHA,str(NATIVE):NATIVE_SHA,str(Path(__file__).resolve()):_sha(__file__),
        str(SESSION/'runtime_sources.json'):_sha(SESSION/'runtime_sources.json')}
    for group in('sources','source_configs'):sources.update(runtime[group])
    sources.update(old.dependency_sources(LEGACY))
    for f,d in sources.items():_pin(f,d)
    c=old._load_clients(LEGACY);old._install_accessibility_compatibility();c.install()
    from optical_ui_v19 import OpticalUI
    from black_pixels import RootPixels
    roles=c.read_roles()
    if roles!=ref['roles']:raise RuntimeError('original display roles changed')
    windows={r:c.window(v['wid'])for r,v in ref['windows'].items()}
    if any(v['pid']!=PID or not v['viewable']for v in windows.values()):raise RuntimeError('original windows missing')
    if not c.inside(windows['control']['rect'],[1080,600,2560,1440])or any(windows[r]['rect']!=ref['windows'][r]['rect']for r in('dmd','slm')):
        raise RuntimeError('original main/output geometry changed')
    before=dict(ref,heartbeat=h,runtime=runtime,roles=roles,windows=windows)
    ui=OpticalUI(SESSION,before);ui._pixels=RootPixels();caches=_caches()
    ui._dmd,ui._phase=[np.load(caches[r]['path'],allow_pickle=False)for r in('dmd','slm')]
    if any(a.shape!=(600,600)or a.dtype!=np.uint8 for a in(ui._dmd,ui._phase))or not np.isin(ui._dmd,(0,255)).all():raise ValueError('exact canonical600/binary DMD required')
    out=Path(out)
    if not out.is_absolute()or out.parent!=ROOT or out.is_symlink():raise ValueError('new direct E038 child required')
    ui._current_profile=dict(schema='optical-lab-current-public-profile-1',active=active,settings=s,caches=caches,roles=roles,sources=sources,
        **{k:h[k]for k in('actual_roi','optical_reference','geometry_frozen')},old_reference_is_window_identity_only=True,control_qualified=False)
    _guard(ui,c);out.mkdir(exist_ok=False);(out/'raw').mkdir();_write(out/'PROFILE.json',ui._current_profile)
    ui.root=out;ui._archives={};ui._last=(-1,-1);ui._zero_attempted=False
    spec=importlib.util.spec_from_file_location('E038_frozen_native',NATIVE);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    return ui,c,native


def _frame(raw,meta,p,revision,requested,last,now):
    import numpy as np
    f=meta['frame'];order=(f.get('TimeStamp'),f.get('logical_sequence'));began=f.get('host_get_started_unix_ns');received=f.get('host_received_unix_ns')
    if (raw.dtype!=np.uint8 or raw.shape!=(554,552)or meta.get('camera_active')!=p['active']or meta.get('settings')!=p['settings']or meta.get('display_revision')!=revision
            or any(type(v)is not int for v in(*order,began,received))or not all(a>b for a,b in zip(order,last))
            or not requested-1_500_000_000<=began<=received<=now or f.get('IsIncomplete')is True
            or any(f.get(k)!=v for k,v in dict(Width=552,Height=554,Stride=552,BufferSize=552*554,pixel_format='Mono8').items())):
        raise RuntimeError('raw format/order/freshness/profile/revision failed; archive retained')
    return order


def snapshot(ui,c,native,out):
    """Save one original single frame, including saturated raw; never arm."""
    import numpy as np
    from PIL import Image
    from optical_lab import legacy_gui as old
    out=Path(out)
    if out!=ui.root:raise ValueError('prepared output differs')
    h,cd,cs=_guard(ui,c);revision=h['display_revision'];prior=set(SESSION.glob('snapshot_*'));requested=time.time_ns()
    if h['last_verified_revision']!=revision:raise RuntimeError('current display revision unverified')
    click=native.invoke(ui,'snapshot',execute=True);end=time.monotonic()+8
    while True:
        current,_,_=_guard(ui,c);fresh=set(SESSION.glob('snapshot_*'))-prior
        if current['display_revision']!=revision or current['last_verified_revision']!=revision or len(fresh)>1:raise RuntimeError('revision/competing snapshot changed')
        if fresh and all((next(iter(fresh))/f).is_file()for f in old.NUMERIC_CORE):
            try:_read(next(iter(fresh))/'metadata.json')
            except(ValueError,OSError):pass
            else:break
        if time.monotonic()>end:raise TimeoutError('one complete new snapshot absent; no retry')
        time.sleep(.04)
    archive=old._copy_snapshot(next(iter(fresh)),out,ui._archives);n=len(ui._archives);_write(out/('ARCHIVE_%03d.json'%n),archive)
    files=archive['files'];raw=np.load(out/files['CCD_raw.npy']['path'],allow_pickle=False);meta=_read(out/files['metadata.json']['path'])
    ui._last=_frame(raw,meta,ui._current_profile,revision,requested,ui._last,time.time_ns())
    if any(not np.array_equal(np.asarray(Image.open(out/files[r+'_canonical_requested.png']['path'])),a)for r,a in(('dmd',cd),('slm',cs))):raise RuntimeError('saved canonical differs; archive retained')
    _guard(ui,c);result=dict(archive=archive,click=click,frame=meta['frame'],raw_max_ADU=int(raw.max()),raw_mean_ADU=float(raw.mean()),saturated_count=int(np.count_nonzero(raw==255)),saturation_rejects_archive=False,control_qualified=False)
    _write(out/('SNAPSHOT_%03d.json'%n),result);return result


def zero(ui,c,native,out):
    """One explicit native allzero attempt, then independent RGB0 evidence."""
    if Path(out)!=ui.root or ui._zero_attempted:raise RuntimeError('prepared output/one zero attempt required')
    _guard(ui,c);ui._zero_attempted=True;result=dict(passed=False,action_attempts=1)
    try:
        result['click']=native.invoke(ui,'allzero',execute=True);end=time.monotonic()+6
        while True:
            h=_heartbeat(ui.state(allow_fault=True),ui._current_profile['active'],ui._current_profile['settings'])
            if h['projection_armed']is False and all(ui._pixels.inspect(tuple(ui.before['windows'][r]['rect']))['all_RGB_zero']for r in('dmd','slm')):result['passed']=True;break
            if time.monotonic()>end:raise TimeoutError('original outputs not verified RGB0')
            time.sleep(.05)
        _guard(ui,c);return result
    except BaseException as e:result['error']=repr(e);raise
    finally:_write(Path(out)/'ZERO_RESULT.json',result)
