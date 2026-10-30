"""Rebuilt-platform experiment. Offline by default; explicit existing-owner I/O.

No SDK, no injected GUI code, no restart, no second camera owner.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
LIVE = LAB / 'experiments/E081_readout_regions_20261028_1'
SESSION = LIVE / 'GUI'
CLIENT = LAB / 'experiments/E083_joint_readout_heads_20261028_1/capture.py'
CLIENT_SHA = '381e02fb516b2b80198f0e5fade5681f4e123f3b3d84dc44e429908a868d27b4'
ASSETS = ROOT/'ASSETS_02'


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_contract(heartbeat):
    return dict(camera_actual=heartbeat['camera_active_readback'],
                roi=heartbeat['actual_roi'], slm=heartbeat['manual_settings'],
                optical_reference=heartbeat['optical_reference'],
                orientation=heartbeat['simulation_display_orientation'])


def validate_current(heartbeat, expected, *, allow_partial_publication=False):
    require(not heartbeat.get('hardware_fault') and heartbeat.get('live_camera'), 'camera/fault state')
    require(heartbeat.get('geometry_frozen'), 'current geometry must stay frozen')
    fields = dict(camera_actual='camera_active_readback', roi='actual_roi', slm='manual_settings',
                  optical_reference='optical_reference', orientation='simulation_display_orientation')
    # The original layered heartbeat writer briefly publishes its base layer
    # before adding camera readback. Missing readback permits READ-ONLY waiting,
    # never completion or UI actions; every already-present field is checked.
    for target, source in fields.items():
        if source not in heartbeat:
            require(allow_partial_publication and source == 'camera_active_readback',
                    'incomplete current parameter publication: '+source)
        else:
            require(heartbeat[source] == expected[target], 'rebuilt platform parameters changed: '+source)
    return 'camera_active_readback' in heartbeat


def make_client(out, expected, maintenance=False):
    """Bind old public transport to explicitly saved NEW parameters, not old data."""
    require(ref(CLIENT)['sha256'] == CLIENT_SHA, 'pinned public client changed')
    sys.path[:0] = [str(CLIENT.parent), str(LAB/'src')]
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi
    Atspi.set_timeout(800, 0)
    old = load(CLIENT, 'e105_existing_owner_transport')

    class Client(old.Client):
        export_ready = None
        export_started = None

        def independent_zero(self):
            proc=Path('/proc',str(old.PID))
            require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==old.START, 'owner lifetime changed')
            cmd=read(LIVE/'LAUNCH_PLAN.json')['command']
            require([x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==cmd[cmd.index('--')+1:],
                    'owner command changed')
            self.c.same_signals(self.roles,self.c.read_roles())
            from black_pixels import RootPixels
            pixels=RootPixels(); result={}
            try:
                for role in ('dmd','slm'):
                    row=self.c.window(self.windows[role]['wid'])
                    require(all(row[k]==self.windows[role][k] for k in ('pid','wid','rect','viewable')), 'output identity changed')
                    result[role]=pixels.inspect(tuple(row['rect']))
                    require(result[role]['all_RGB_zero'], 'actual framebuffer nonzero')
            finally:pixels.close()
            return result

        def wait_export(self):
            # Read-only recovery; never authorize actions on stale owner state.
            initial=read(SESSION/'heartbeat.json'); begin=time.monotonic(); trace=[]
            deadline=min(begin+45,self.ui.deadline-40)
            while time.monotonic()<deadline:
                require((self.export_ready is not None and self.export_ready()) or
                        (self.export_started is not None and self.export_started()),
                        'no current complete sampling/EXPORT evidence')
                zero=self.independent_zero(); h=read(SESSION/'heartbeat.json')
                complete=validate_current(h,expected,allow_partial_publication=True)
                require(h['pid']==old.PID and h['unix_ns']>=initial['unix_ns'], 'owner publication changed/regressed')
                require(not any(h.get(k) for k in ('loop_active','deployment_busy','adaptation_active'))
                        and not h.get('spatial_calibration',{}).get('busy'), 'competing task')
                require(h['projection_armed'] is False or h['unix_ns']==initial['unix_ns'], 'new publication still armed')
                age=(time.time_ns()-h['unix_ns'])/1e9; f=h.get('latest_frame') or {}
                fa=f.get('age_seconds');trace.append(dict(publication=h['unix_ns'],age=age,frame=f.get('frame_id')))
                if (complete and h['unix_ns']>initial['unix_ns'] and h['projection_armed'] is False and 0<=age<3
                        and f.get('frame_id')!=initial.get('latest_frame',{}).get('frame_id')
                        and isinstance(fa,(int,float)) and math.isfinite(fa) and 0<=fa and fa+age<=.75):
                    self.independent_zero()
                    self.record('READ_ONLY_EXPORT_WAIT',dict(trace=trace,seconds=time.monotonic()-begin,zero=zero,
                        no_clicks=True,not_completion_or_projection_authority=True))
                    return
                time.sleep(.1)
            raise TimeoutError('complete sampling EXPORT did not regain fresh owner within bounded read-only wait')

        def state(self, allow_fault=False):
            # Original process identity/lifetime/cmdline and competition guards.
            deadline = time.monotonic()+.5
            while True:
                try:
                    h = super().state(allow_fault=True)
                except RuntimeError as error:
                    if str(error)!='stale or changed owner' or self.export_ready is None:
                        raise
                    self.wait_export()
                    h=super().state(allow_fault=True)
                if maintenance:
                    require(h.get('hardware_fault') in (None,'硬触发故障：duplicate/out-of-order/reset camera frame'),
                            'unexpected fault; maintenance stopped')
                    require(h['projection_armed'] is False,'maintenance must remain disarmed')
                    for key,target in (('actual_roi','roi'),('manual_settings','slm'),('optical_reference','optical_reference'),
                                       ('simulation_display_orientation','orientation')):
                        require(h[key]==expected[target],'retained configuration changed during maintenance')
                    self.independent_zero()
                    return h
                if validate_current(h,expected,allow_partial_publication=True):
                    break
                if self.export_started is not None and self.export_started():
                    self.wait_export()
                    continue
                require(time.monotonic() < deadline, 'incomplete camera publication persisted')
                time.sleep(.01)
            validate_current(h, expected)
            return h

        def owner_read(self):
            return self.state()

        def click(self,role,name):
            if maintenance:
                require((role,name) in (('page tab','主界面 · 光路与参数'),
                    ('push button','复核并恢复（不投图）'),('push button','仅CCD（硬触发）')),
                    'maintenance action outside exact whitelist')
                self.state()
                if role=='page tab':
                    # A stopped camera cannot satisfy the acquisition client's
                    # fresh-frame condition. Selecting ONLY the main tab is a
                    # zero-output maintenance action, not a projection bypass.
                    node,A=self.one(role,name);runtime=self.native._runtime()
                    wid=self.native._wid(self.ui.main['wid'])
                    rect=self.pub._rect(node,A,runtime,self.ui,role)
                    self.native._same_window(runtime,wid,self.ui.pid,self.ui.main['rect'])
                    self.native._reject_other_dialogs(self.ui.pid,wid)
                    cls=runtime.Pointer
                    require(cls.__name__=='NumLockPointer' and cls.__module__=='workbench.unified.pointer_control',
                            'original pointer guard required')
                    focus=runtime.Input(wid);pointer=None
                    try:
                        require(self.native._active_window()==wid,'maintenance focus differs')
                        self.state();self.native._reject_other_dialogs(self.ui.pid,wid)
                        require(self.pub._rect(node,A,runtime,self.ui,role)==rect,'maintenance tab moved')
                        tree=self.native._root_child(focus,wid);pointer=cls()
                        require(self.native._active_window()==wid,'maintenance focus changed')
                        pointer.click((rect[0]+rect[2]//2,rect[1]+rect[3]//2),dict(wid=tree['pointer_root_child_wid']))
                        self.record('ZERO_OUTPUT_MAIN_TAB',dict(name=name,rect=rect,root_parent_proof=tree,
                            no_projection=True,no_new_window=True))
                    finally:
                        if pointer is not None:pointer.close()
                        focus.close()
                    return
            return super().click(role,name)

    client = Client(out)
    client.expected = expected
    client.ui.root = out
    original_deps = client.pub._deps

    def deps(namespace):
        value = original_deps(namespace)
        value.native.ORDINARY = list(client.native.ORDINARY)
        return value

    client.pub._deps = deps
    return client


def helper():
    root = LAB/'experiments/E094_completion_wait_20261029_1'
    require(ref(root/'run.py')['sha256'] == 'ac8657b143ceccc85b6080cbac3f3c4028eb23e262cbfd8ca139fcd3ed890dc8',
            'completion helper changed')
    sys.path.insert(0, str(root))
    return load(root/'run.py', 'e105_completion_helpers')


def prepare_assets(output):
    """File-only pilot assets. Keep applied R2, not the unapplied R4 widget."""
    import numpy as np
    baseline = read(ROOT/'BASELINE_01/BASELINE.json')
    catalog_ref = baseline['ui_events']['RAW_validation_catalog_loaded']['reference']
    require(ref(catalog_ref['path']) == catalog_ref, 'baseline catalog changed')
    catalog = read(catalog_ref['path'])
    catalog['inputs'] = [row for row in catalog['inputs']
                         if row.get('input_kind') != 'calibration_probe_not_MuJoCo_state'
                         and not row.get('state_fields_are_loader_placeholders')]
    require(len(catalog['inputs'])==9, 'nine explicit state inputs required')
    source = baseline['files']['slm']
    require(ref(source['path']) == source, 'baseline mask changed')
    base = np.load(source['path'], allow_pickle=False)
    require(base.shape == (600,600) and base.dtype == np.uint8, '600-square mask required')
    macro = base[::8, ::8]
    require(np.array_equal(base, macro.repeat(8,0).repeat(8,1)), 'baseline is not exact 8x8 macro pixels')
    output.mkdir()
    catalogs = {}
    for name, ids in [('SMOKE', ['CENTER', 'VY_POS']), ('PILOT', [r['input_id'] for r in catalog['inputs']])]:
        selected = dict(catalog, inputs=[r for i in ids for r in catalog['inputs'] if r['input_id']==i])
        require(len(selected['inputs']) == len(ids), 'unique existing inputs required')
        for row in selected['inputs']:
            require(ref(row['dmd']['path']) == row['dmd'], 'input changed')
        write(output/(name+'_CATALOG.json'), selected)
        catalogs[name] = ref(output/(name+'_CATALOG.json'))
    yy, xx = np.meshgrid((np.arange(75)+.5)/75, (np.arange(75)+.5)/75, indexing='ij')
    bases = np.array([np.sin(2*np.pi*4*xx), np.sin(2*np.pi*4*yy),
                      np.sin(2*np.pi*8*xx), np.sin(2*np.pi*8*yy)], dtype=np.float32)
    with (output/'BASES.npy').open('xb') as f:
        np.save(f, bases, allow_pickle=False)
    patterns = [('BASE', base, np.zeros(4), 0.)]
    for k, axis in enumerate(('X4','Y4','X8','Y8')):
        for sign, label in [(1,'P'),(-1,'M')]:
            coeff = np.zeros(4); coeff[k] = sign*8
            unbounded = macro.astype(float)+np.einsum('k,khw->hw', coeff, bases)
            patch = np.clip(np.rint(unbounded),0,255).astype(np.uint8).repeat(8,0).repeat(8,1)
            patterns.append((axis+label, patch, coeff, float(np.mean((unbounded<0)|(unbounded>255)))))
    assets = {}
    for name, patch, coeff, clip_fraction in patterns:
        file = output/(name+'.npy')
        with file.open('xb') as f:
            np.save(f, patch, allow_pickle=False)
        assets[name] = dict(patch=ref(file), coefficients=coeff.tolist(),
            changed_fraction=float(np.mean(patch!=base)), gray_clip_fraction=clip_fraction,
            max_gray_delta=int(np.abs(patch.astype(int)-base.astype(int)).max()))
    write(output/'PLAN.json', dict(baseline=ref(ROOT/'BASELINE_01/BASELINE.json'),
        original_catalog=catalog_ref, catalogs=catalogs, assets=assets, bases=ref(output/'BASES.npy'),
        condition_scope='new platform, 4 directions, ±8 raw gray; NOT calibrated radians',
        pilot_inputs=9, pilot_masks=9, formal16_not_authorized_by_pilot=True,
        source=ref(__file__), hardware_actions=0))
    print('Offline assets prepared: 9 masks, original R2 inputs.', flush=True)


def verify_test_plan(actual, contract, baseline, catalog, mask, frames, repeats):
    import numpy as np
    settings = actual['settings']
    require((settings['frames'],settings['filter_mode'],settings['repeats'],settings['settle_seconds'])
            == (frames,'mean',repeats,1.5), 'sampling parameters differ')
    rows = read(catalog['path'])['inputs']; ids = [r['input_id'] for r in rows]
    require(settings['input_ids']==ids and actual['input_sources']=={r['input_id']:r['dmd'] for r in rows},
            'input binding differs')
    require(contract['input_catalog']['reference']==catalog, 'catalog reference differs')
    require(contract['expected_active']==baseline['expected']['camera_actual'], 'camera differs')
    require(contract['dmd_presentation']['side']==600, 'DMD scale changed')
    main = contract['research_test']['shared_main']
    require(main['roi']==baseline['expected']['roi'] and main['slm']==baseline['expected']['slm'],
            'placement changed')
    require(len(contract['research_test']['layout']['regions'])==64, 'readout count changed')
    slm = baseline['expected']['slm']
    panel = np.zeros((1080,1920), np.uint8)
    x, y = 660-slm['x'], 240+slm['y']
    panel[y:y+600,x:x+600] = np.load(mask['path'], allow_pickle=False)
    actual_mask = contract['fixed_slm']
    require(ref(actual_mask['path'])==actual_mask, 'sealed mask bytes changed')
    require(np.array_equal(np.load(actual_mask['path'],allow_pickle=False), panel), 'actual full-panel mask differs')
    require(actual['accepted_RAW_budget']==len(ids)*frames*repeats, 'RAW budget differs')


def capture_batch(output, catalog_name, mask_name, frames, repeats, authorization, assets=ASSETS):
    require(authorization.strip(), 'explicit current authorization required')
    baseline = read(ROOT/'BASELINE_01/BASELINE.json')
    require(assets.parent==ROOT,'explicit experiment asset directory required')
    plan = read(assets/'PLAN.json')
    require(plan['baseline']==ref(ROOT/'BASELINE_01/BASELINE.json'),'asset platform baseline differs')
    catalog = plan['catalogs'][catalog_name]; mask = plan['assets'][mask_name]['patch']
    require(ref(catalog['path'])==catalog and ref(mask['path'])==mask, 'asset bytes changed')
    require(2<=frames<=120 and 3<=repeats<=5, 'bounded frames/repeats; original GUI requires at least 3 visits')
    require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'], '10h authorization expired')
    output.mkdir(); start=time.monotonic(); deadline=start+300
    write(output/'REQUEST.json', dict(catalog=catalog, mask=mask, frames=frames, repeats=repeats,
        assets_plan=ref(assets/'PLAN.json'),mask_name=mask_name,coefficients=plan['assets'][mask_name]['coefficients'],
        baseline=plan['baseline'], source=ref(__file__), authorization=authorization,
        max_seconds=300, no_restart=True, no_camera_writes=True, no_closed_loop=True))
    client=None; active=False; folder=None; probe=None
    try:
        client=make_client(output,baseline['expected']); client.ui.deadline=deadline
        require(not client.state()['projection_armed'], 'must begin disarmed')
        client.zeros()
        client.tab('主界面 · 光路与参数')
        node,A=client.one('spin button','有效帧RAW数N')
        require(node.get_value_iface().set_current_value(frames), 'N setting rejected')
        require(node.get_value_iface().get_current_value()==frames, 'N readback differs')
        loaded=client.pub.load_live(client.ui,client.c,'slm',Path(mask['path']))
        client.record('LOADED_MASK',loaded)
        client.zeros();client.tab('读出稳定性测试');client.load_catalog(catalog)
        client.configure_test()
        nodes,A=client.nodes('spin button');nodes=[n for n in nodes if n.get_name()=='输入激励与采样']
        import optical_ui_v19 as optical
        nodes.sort(key=lambda n:(optical.extents(n,A)[1],optical.extents(n,A)[0]))
        require(len(nodes)==3, 'test controls ambiguous')
        require(nodes[0].get_value_iface().set_current_value(repeats), 'repeats rejected')
        require(nodes[0].get_value_iface().get_current_value()==repeats,'repeat readback differs; do not start')
        e=helper(); prior=set((LIVE/'TESTS').glob('TEST_*')); seal=e.ExportSealReader()
        probe=e.cache_start_button(client,deadline-40); verified=False; last=-1
        active=True;e.click_cached_start(client,probe)
        def result_reader():
            nonlocal folder,verified,last
            created=set((LIVE/'TESTS').glob('TEST_*'))-prior
            require(len(created)<=1, 'competing test')
            if not created:return None
            folder=next(iter(created))
            for name in ('FAILED.json','START_FAILED.json'):
                if (folder/name).exists():raise RuntimeError('GUI test failed: '+str(folder/name))
            if not verified and (folder/'PLAN.json').exists():
                actual=read(folder/'PLAN.json'); contract=read(actual['contract']['path'])
                require(ref(actual['contract']['path'])==actual['contract'], 'contract bytes changed')
                verify_test_plan(actual,contract,baseline,catalog,mask,frames,repeats)
                client.record('ACTUAL_PLAN',actual); verified=True
            count=len(list(folder.glob('EFFECTIVE_*/READOUT.json')))
            if count!=last:
                print(mask_name,catalog_name,'effective frames',count,flush=True);last=count
            sealed=seal(folder)
            if sealed is None:return None
            value,refs=sealed
            require(verified and value['all_visits_completed'] and value['final_actual_zero_verified'], 'incomplete result')
            expected=len(read(catalog['path'])['inputs'])*frames*repeats
            require(value['accepted_raw_total']==expected, 'result raw count differs')
            return dict(**refs,raw_frames=expected,test_plan=ref(folder/'PLAN.json'))
        def export_ready():
            if folder is None or not verified:return False
            require(not (folder/'FAILED.json').exists() and not (folder/'START_FAILED.json').exists(), 'test failed during export')
            if len(list(folder.glob('EFFECTIVE_*/READOUT.json')))!=len(read(catalog['path'])['inputs'])*repeats:return False
            path=folder/'RESULTS/RESULT.json'
            if not path.is_file():return False
            try:value=read(path)
            except json.JSONDecodeError:return False
            return (value.get('all_visits_completed') is True and value.get('final_actual_zero_verified') is True
                    and value.get('accepted_raw_total')==len(read(catalog['path'])['inputs'])*frames*repeats)
        def export_started():
            if folder is None or not verified:return False
            require(not (folder/'FAILED.json').exists() and not (folder/'START_FAILED.json').exists(), 'test failed during export')
            # This evidence permits waiting ONLY. Completion still requires the
            # sealed RESULT, live controls and fresh exact readback barrier.
            return len(list(folder.glob('EFFECTIVE_*/READOUT.json')))==len(read(catalog['path'])['inputs'])*repeats
        client.export_ready=export_ready
        client.export_started=export_started
        proof=e.barrier(client,result_reader,deadline-time.monotonic()-40,probe=probe)
        client.export_ready=None
        client.export_started=None
        active=False
        client.tab('主界面 · 光路与参数')
        restored=client.pub.load_live(client.ui,client.c,'slm',Path(baseline['files']['slm']['path']))
        zero=client.zeros()
        require(time.monotonic()<deadline, 'batch exceeded 300s including restore')
        write(output/'RESULT.json',dict(completed=True,receipt=proof['result'],completion=proof,
            restored_mask=restored,zero=zero,elapsed_seconds=time.monotonic()-start,
            final_heartbeat=client.state(),catalog_left_loaded=catalog,N_left_applied=frames))
        print('BATCH COMPLETE',proof['result']['raw_frames'],'RAW',round(time.monotonic()-start,2),'s',flush=True)
    except BaseException as error:
        cleanup={}
        if client is not None:
            try:
                if active:
                    client.click('push button','停止测试／全0')
                    time.sleep(.5)
                cleanup['zero']=park(client)
            except BaseException as problem:
                cleanup['error']=repr(problem)
        write(output/'FAILED.json',dict(error=repr(error),folder=None if folder is None else str(folder),
            cleanup=cleanup,elapsed_seconds=time.monotonic()-start,no_auto_restart=True))
        raise


def save_snapshot(client):
    prior = set(SESSION.glob('snapshot_*'))
    client.click('push button', '保存当前对照')
    end = time.monotonic()+15
    while time.monotonic() < end:
        client.state()
        new = set(SESSION.glob('snapshot_*'))-prior
        require(len(new) <= 1, 'competing snapshot')
        if new:
            folder = next(iter(new))
            needed = ['metadata.json', 'comparison.json', 'CCD_raw.npy',
                      'dmd_canonical_requested.png', 'slm_canonical_requested.png']
            if all((folder/n).is_file() for n in needed):
                try:
                    metadata, comparison = read(folder/'metadata.json'), read(folder/'comparison.json')
                except json.JSONDecodeError:
                    time.sleep(.1)
                    continue
                require(metadata['camera_active'] == client.expected['camera_actual'], 'snapshot camera differs')
                return dict(folder=str(folder), files={n:ref(folder/n) for n in needed},
                            geometry=comparison['geometry'])
        time.sleep(.1)
    raise TimeoutError('current snapshot did not complete')


def park(client):
    client.click('push button', '两路全0，CCD保持（Esc）')
    time.sleep(.4)
    return client.zeros()


def background_snapshots(output,authorization):
    """Zero-output diagnostic snapshots, NOT shutter-dark or N20 training data."""
    require(authorization.strip(),'explicit current authorization required')
    require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'],'authorization expired')
    output.mkdir();b=read(ROOT/'BASELINE_01/BASELINE.json');client=make_client(output,b['expected'])
    client.ui.deadline=time.monotonic()+120;client.zeros();client.tab('主界面 · 光路与参数')
    snapshots=[];last=None
    for i in range(5):
        before=client.zeros();h=client.state();snap=save_snapshot(client);after=client.zeros()
        meta=read(snap['files']['metadata.json']['path']);f=meta['frame']
        require(f['host_get_started_unix_ns']>h['unix_ns']-int(1e9),'snapshot stale')
        require(last is None or f['host_received_unix_ns']>last,'snapshot repeated')
        last=f['host_received_unix_ns']
        snapshots.append(dict(before_zero=before,after_zero=after,snapshot=snap))
        time.sleep(.2)
    write(output/'RESULT.json',dict(snapshots=snapshots,source=ref(__file__),authorization=authorization,
        diagnostic_ONLY=True,shutter_dark_NOT_measured=True,N20_effective_frame=False,
        manual_preview_quantitative_valid_NOT_overridden=True,training_data=False,no_projection=True,zero=client.zeros()))


def restore_baseline(output, authorization):
    require(authorization.strip(), 'explicit current authorization required')
    require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'], 'authorization expired')
    output.mkdir(); b=read(ROOT/'BASELINE_01/BASELINE.json')
    client=make_client(output,b['expected']);client.ui.deadline=time.monotonic()+120
    require(not client.state()['projection_armed'], 'restore must begin disarmed')
    before=client.zeros();client.tab('主界面 · 光路与参数')
    source=b['files']['slm'];require(ref(source['path'])==source,'baseline mask changed')
    restored=client.pub.load_live(client.ui,client.c,'slm',Path(source['path']))
    write(output/'RESULT.json',dict(before=before,mask=restored,after=client.zeros(),
        heartbeat=client.state(),no_projection=True,no_camera_writes=True,source=ref(__file__),authorization=authorization))


def restore_session_controls(output,authorization):
    """Return only experiment-changed UI controls/caches, in existing zero windows.

    Do not apply the pending R4 ball control: baseline actually used cached R2.
    This does not stop/start the camera or alter its restoration transaction.
    """
    require(authorization.strip(),'explicit existing-owner authority required')
    require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'],'authorization expired')
    output.mkdir();b=read(ROOT/'BASELINE_01/BASELINE.json');client=None
    try:
        client=make_client(output,b['expected']);client.ui.deadline=time.monotonic()+180
        require(not client.state()['projection_armed'],'control restore requires disarmed outputs')
        before=client.zeros();client.tab('读出稳定性测试')
        node,A=client.one('push button','一键测试')
        require(node.get_state_set().contains(A.StateType.ENABLED),'cannot restore during active test')
        catalog=b['ui_events']['RAW_validation_catalog_loaded']['reference']
        require(ref(catalog['path'])==catalog,'baseline catalog bytes changed')
        expected_ids=b['ui_events']['RAW_validation_catalog_loaded']['input_ids']
        state_ids=[r['input_id'] for r in read(catalog['path'])['inputs']
            if r.get('input_kind')!='calibration_probe_not_MuJoCo_state'
            and not r.get('state_fields_are_loader_placeholders')]
        require(state_ids==expected_ids,'original GUI-filtered state IDs changed')
        # A previous normal load can have succeeded in the GUI but failed in
        # the legacy external client's all-input count (it includes REF_SIN).
        # Resume only from an exact recorded catalog acknowledgement; do not
        # suppress a load exception, fabricate an event, or load a new file.
        events,_=event_state();ack=events.get('RAW_validation_catalog_loaded',{})
        if ack.get('reference')==catalog and ack.get('input_ids')==expected_ids:
            client.record('EXACT_ORIGINAL_CATALOG_ALREADY_ACKNOWLEDGED',ack)
        else:
            client.load_catalog(catalog)
        events,_=event_state();ack=events.get('RAW_validation_catalog_loaded',{})
        require(ack.get('reference')==catalog and ack.get('input_ids')==expected_ids,
                'restored original catalog acknowledgement differs')
        client.tab('主界面 · 光路与参数')
        restored={}
        for kind in ('dmd','slm'):
            source=b['files'][kind];require(ref(source['path'])==source,'baseline input cache changed')
            restored[kind]=client.pub.load_live(client.ui,client.c,kind,Path(source['path']))
            client.zeros()
        filt=b['ui_events']['shared_effective_filter_controls']['controls']
        require(filt['mode']=='mean','only unchanged mean-mode baseline supported')
        node,A=client.one('spin button','有效帧RAW数N')
        require(node.get_value_iface().set_current_value(filt['frames']),'baseline N restore rejected')
        require(node.get_value_iface().get_current_value()==filt['frames'],'baseline N readback differs')
        # Qt publishes the value-change event asynchronously. Read-only wait;
        # never reapply parameters or treat an absent event as success.
        event_deadline=min(client.ui.deadline,time.monotonic()+5)
        while True:
            events,files=event_state()
            if (files==b['files'] and
                events.get('RAW_validation_catalog_loaded',{}).get('reference')==catalog and
                events.get('shared_effective_filter_controls',{}).get('controls',{}).get('frames')==filt['frames']):
                break
            if time.monotonic()>=event_deadline:break
            time.sleep(.05)
        require(files==b['files'],'restored DMD/SLM cache differs')
        require(events['RAW_validation_catalog_loaded']['reference']==catalog,'catalog restoration differs')
        require(events['shared_effective_filter_controls']['controls']['frames']==filt['frames'],'filter event differs')
        write(output/'RESULT.json',dict(before=before,restored=restored,original_catalog=catalog,N=filt['frames'],
            final_zero=client.zeros(),heartbeat=client.state(),source=ref(__file__),
            pending_ball_radius_NOT_applied=True,no_new_projection=True,no_camera_writes=True,no_restart=True))
    except BaseException as error:
        cleanup={}
        if client is not None:
            try:cleanup=park(client)
            except BaseException as problem:cleanup=dict(error=repr(problem))
        write(output/'FAILED.json',dict(error=repr(error),cleanup=cleanup,no_restart=True));raise


def recheck_camera(output,authorization):
    """One ordinary existing-window recheck and camera-only start, never retry.

    Does not modify trigger/counter code or suppress its faults. Same requested
    acquisition settings are reapplied by the GUI's original driver transaction.
    """
    require(authorization.strip(),'explicit existing-window authority required')
    require(time.time_ns()<read(ROOT/'BASELINE_01/AUTHORIZATION.json')['end_unix_ns'],'authorization expired')
    output.mkdir();b=read(ROOT/'BASELINE_01/BASELINE.json');start=time.monotonic()
    report=SESSION/'live_20261029_213025_905881/report.json';oldreport=read(report)
    require(oldreport['camera_settings_restored'] and oldreport['restore_differences']=={},'original camera rollback failed')
    require(oldreport['display_guard']['error'] is None,'display fault forbids recovery')
    write(output/'BEFORE.json',dict(report=ref(report),report_copy=oldreport,heartbeat=read(SESSION/'heartbeat.json'),
        source=ref(__file__),authorization=authorization,no_window_restart=True,no_projection=True))
    client=make_client(output,b['expected'],maintenance=True);client.ui.deadline=start+150
    try:
        require(not client.state()['live_camera'],'original camera must already be released')
        client.tab('主界面 · 光路与参数');client.dump_controls()
        prior=set(SESSION.glob('recheck_*'));client.click('push button','复核并恢复（不投图）')
        while time.monotonic()<start+90:
            h=client.state()
            if not h['hardware_fault'] and not h['busy']:break
            time.sleep(.2)
        else:raise TimeoutError('original read-only recheck did not pass')
        checks=set(SESSION.glob('recheck_*'))-prior;require(len(checks)==1,'one original recheck required')
        client.record('RECHECK_COMPLETE',dict(folder=str(next(iter(checks))),heartbeat=h,zero=client.zeros()))
        client.click('push button','仅CCD（硬触发）')
        last=None;fresh=0
        while time.monotonic()<start+140:
            h=client.state()
            if h.get('live_camera') and h.get('geometry_frozen') and 'camera_active_readback' in h and h.get('latest_frame'):
                validate_current(h,b['expected']);f=h['latest_frame']
                if f.get('frame_id')!=last and f.get('age_seconds',999)<.5:fresh+=1;last=f['frame_id']
                if fresh>=5:
                    write(output/'RESULT.json',dict(restored=True,zero=client.zeros(),heartbeat=h,seconds=time.monotonic()-start,
                        exact_new_platform_parameters_restored=True,new_camera_epoch=True,
                        fault_not_disabled=True,no_window_restart=True,no_new_projection=True))
                    return
            time.sleep(.3)
        raise TimeoutError('camera-only same-parameter fresh-frame check did not pass')
    except BaseException as error:
        write(output/'FAILED.json',dict(error=repr(error),heartbeat=read(SESSION/'heartbeat.json'),
            no_retry=True,no_new_window=True,no_projection=True))
        raise


def event_state():
    """Replay declared UI events, not mtime-based artifact selection."""
    data = (SESSION/'events.jsonl').read_bytes()
    require(data.endswith(b'\n'), 'partial event publication')
    result = {}
    files = {}
    for line in data.splitlines():
        row = json.loads(line)
        state = row.get('state')
        result[state] = row
        if state == 'manual_file_loaded':
            files['slm' if row['SLM'] else 'dmd'] = {k:row[k] for k in ('path','sha256')}
    require(set(files) == {'dmd','slm'}, 'both actual file caches required')
    for item in files.values():
        require(ref(item['path']) == item, 'cached source changed')
    return result, files


def baseline(output, authorization):
    require(authorization.strip(), 'explicit current authorization required')
    output.mkdir()
    h = read(SESSION/'heartbeat.json')
    expected = camera_contract(h)
    write(output/'AUTHORIZATION.json', dict(authorization=authorization, unix_ns=time.time_ns(),
        end_unix_ns=time.time_ns()+10*3600*10**9, scope='existing owner, bounded static batches <=300s, zero after each',
        no_restart=True, no_real_actuators=True, source=ref(__file__)))
    write(output/'EXPECTED.json', expected)
    write(output/'BEFORE_HEARTBEAT.json', h)
    client = None
    try:
        client = make_client(output, expected)
        require(client.state()['current_page'] == 0, 'baseline expects current main page')
        snapshot = save_snapshot(client)
        write(output/'SNAPSHOT.json', snapshot)
        geometry_path = Path(snapshot['geometry']['reference_folder'])/'frozen_geometry.json'
        require(read(geometry_path) == snapshot['geometry'], 'snapshot/current frozen geometry differ')
        zero = park(client)
        main = client.dump_controls()
        events, files = event_state()
        write(output/'CACHE.json', files)
        selected = {k:events[k] for k in ('RAW_validation_catalog_loaded', 'shared_effective_filter_controls',
            'manual_display_controls', 'dmd_scale_changed', 'readout_layout_changed') if k in events}
        write(output/'UI_EVENTS.json', selected)
        client.tab('读出稳定性测试')
        controls = client.dump_controls()
        client.tab('主界面 · 光路与参数')
        write(output/'BASELINE.json', dict(expected=expected, snapshot=snapshot, geometry=ref(geometry_path),
            files=files, main_controls=main, test_controls=controls, ui_events=selected,
            windows=client.windows, roles=client.roles, identity_pid=client.ui.pid,
            output_zero=zero, final_zero=client.zeros(), final_heartbeat=client.state(),
            no_new_projection=True, no_camera_writes=True, no_restart=True))
        print('BASELINE saved; current NEW platform archived; DMD/SLM actual zero.', flush=True)
    except BaseException as error:
        cleanup = {}
        if client is not None:
            try:
                cleanup = park(client)
            except BaseException as problem:
                cleanup = dict(error=repr(problem), no_restart=True)
        write(output/'FAILED.json', dict(error=repr(error), cleanup=cleanup))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--prepare-assets', action='store_true')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--restore-baseline', action='store_true')
    parser.add_argument('--restore-session-controls',action='store_true')
    parser.add_argument('--recheck-camera',action='store_true')
    parser.add_argument('--background-snapshots',action='store_true')
    parser.add_argument('--catalog', default='SMOKE')
    parser.add_argument('--assets',default='ASSETS_02')
    parser.add_argument('--mask', default='BASE')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--authorization', default='')
    parser.add_argument('--output', default='BASELINE_01')
    args = parser.parse_args()
    if args.prepare_assets:
        require(not args.execute_authorized, 'asset preparation is offline only')
        prepare_assets(ASSETS)
        return
    if not args.execute_authorized:
        print('OFFLINE default: no GUI, camera, USB or display access.')
        return
    require(sum((args.baseline,args.capture,args.restore_baseline,args.restore_session_controls,args.recheck_camera,args.background_snapshots))==1, 'explicit single operation required')
    require(Path(args.output).name == args.output and args.output not in ('.', '..'), 'fresh direct child name required')
    if args.restore_session_controls:
        restore_session_controls(ROOT/args.output,args.authorization)
    elif args.background_snapshots:
        background_snapshots(ROOT/args.output,args.authorization)
    elif args.recheck_camera:
        recheck_camera(ROOT/args.output,args.authorization)
    elif args.restore_baseline:
        restore_baseline(ROOT/args.output,args.authorization)
    elif args.baseline:
        baseline(ROOT/args.output, args.authorization)
    else:
        require(Path(args.assets).name==args.assets,'explicit assets name required')
        capture_batch(ROOT/args.output,args.catalog,args.mask,args.frames,args.repeats,args.authorization,ROOT/args.assets)


if __name__ == '__main__':
    main()
