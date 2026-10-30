import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import workflow as w


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.b = w.read(w.ROOT/'BASELINE_01/BASELINE.json')
        self.h = self.b['final_heartbeat']

    def test_new_platform_not_old_parameters(self):
        w.validate_current(self.h,self.b['expected'])
        self.assertEqual(self.b['expected']['roi']['Width'],552)
        self.assertEqual(self.b['expected']['slm']['x'],32)

    def test_counter_window_scheduling_does_not_accept_wrapped_RAW(self):
        from series import counter_margin
        self.assertTrue(counter_margin(0));self.assertTrue(counter_margin(10000))
        self.assertFalse(counter_margin(60000));self.assertFalse(counter_margin(65535))
        for bad in (-1,65536,None,1.2):
            with self.assertRaises(RuntimeError):counter_margin(bad)

    def test_session_cutoff_never_extends_authority(self):
        from unittest.mock import patch
        from series import capture_end_ns,LAST_BATCH_END_NS
        for authority in (LAST_BATCH_END_NS-1,LAST_BATCH_END_NS+1000):
            with patch('series.read',return_value={'end_unix_ns':authority}):
                self.assertEqual(capture_end_ns(),min(authority,LAST_BATCH_END_NS))

    def test_parameter_and_fault_changes_rejected(self):
        for key, value in [('hardware_fault','error'),('live_camera',False),('geometry_frozen',False)]:
            h=copy.deepcopy(self.h);h[key]=value
            with self.assertRaises(RuntimeError):w.validate_current(h,self.b['expected'])
        for field in ('roi','slm','camera_actual','optical_reference','orientation'):
            expected=copy.deepcopy(self.b['expected']);expected[field]={}
            with self.assertRaises(RuntimeError):w.validate_current(self.h,expected)

    def test_assets_are_local_gray_probes(self):
        p=w.read(w.ASSETS/'PLAN.json')
        base=np.load(self.b['files']['slm']['path'],allow_pickle=False)
        self.assertEqual(len(p['assets']),9)
        for name,item in p['assets'].items():
            self.assertEqual(w.ref(item['patch']['path']),item['patch'])
            mask=np.load(item['patch']['path'],allow_pickle=False)
            self.assertEqual(mask.shape,(600,600));self.assertEqual(mask.dtype,np.uint8)
            self.assertLessEqual(np.abs(mask.astype(int)-base.astype(int)).max(),8)
            np.testing.assert_array_equal(mask,mask[::8,::8].repeat(8,0).repeat(8,1))
            if name=='BASE':np.testing.assert_array_equal(mask,base)
        for item in p['catalogs'].values():
            for row in w.read(item['path'])['inputs']:
                self.assertEqual(row['ball_size']['radius_raw_px'],2)

    def test_partial_export_publication_only_waits(self):
        h=copy.deepcopy(self.h);del h['camera_active_readback']
        with self.assertRaises(RuntimeError):w.validate_current(h,self.b['expected'])
        self.assertFalse(w.validate_current(h,self.b['expected'],allow_partial_publication=True))
        self.assertTrue(w.validate_current(self.h,self.b['expected'],allow_partial_publication=True))
        for field,value in [('hardware_fault','fault'),('actual_roi',{}),('manual_settings',{}),
                            ('geometry_frozen',False),('optical_reference',{})]:
            changed=copy.deepcopy(h);changed[field]=value
            with self.assertRaises(RuntimeError):
                w.validate_current(changed,self.b['expected'],allow_partial_publication=True)
        del h['actual_roi']
        with self.assertRaises(RuntimeError):
            w.validate_current(h,self.b['expected'],allow_partial_publication=True)

    def test_sealed_plan_checks_current_panel_and_raw_budget(self):
        p=w.read(w.ASSETS/'PLAN.json');cat=p['catalogs']['SMOKE'];mask=p['assets']['BASE']['patch']
        rows=w.read(cat['path'])['inputs'];ids=[r['input_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'panel.npy';panel=np.zeros((1080,1920),np.uint8)
            slm=self.b['expected']['slm'];x,y=660-slm['x'],240+slm['y']
            panel[y:y+600,x:x+600]=np.load(mask['path'],allow_pickle=False)
            np.save(file,panel,allow_pickle=False)
            contract=dict(input_catalog=dict(reference=cat),expected_active=self.b['expected']['camera_actual'],
                dmd_presentation=dict(side=600),fixed_slm=w.ref(file),research_test=dict(
                    shared_main=dict(roi=self.b['expected']['roi'],slm=slm),layout=dict(regions=list(range(64)))))
            actual=dict(settings=dict(frames=50,filter_mode='mean',repeats=3,settle_seconds=1.5,input_ids=ids),
                input_sources={r['input_id']:r['dmd'] for r in rows},accepted_RAW_budget=300)
            w.verify_test_plan(actual,contract,self.b,cat,mask,50,3)
            actual['accepted_RAW_budget']=299
            with self.assertRaises(RuntimeError):w.verify_test_plan(actual,contract,self.b,cat,mask,50,3)
            actual['accepted_RAW_budget']=300;contract['dmd_presentation']['side']=1080
            with self.assertRaises(RuntimeError):w.verify_test_plan(actual,contract,self.b,cat,mask,50,3)

    def test_no_hardware_import_on_import(self):
        import subprocess,sys
        code="import sys; import workflow; assert not any(k.startswith(('gi','PyQt','PySpin','spin_c')) for k in sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=w.ROOT,check=True)

    def test_strong_design_frozen_split_and_native_bounds(self):
        p=w.read(w.ROOT/'DESIGN_STRONG_01/PLAN.json');base=np.load(self.b['files']['slm']['path'],allow_pickle=False)
        self.assertEqual(p['condition_pairs'],232);self.assertEqual(p['coefficient_L1_max'],24)
        trajectories={s:set() for s in ('TRAIN','SELECT','TEST')};ids={s:set() for s in trajectories}
        for job in p['jobs']:
            catalog=w.read(p['catalogs'][job['catalog']]['path'])
            self.assertGreaterEqual(len(catalog['inputs']),2);self.assertLessEqual(len(catalog['inputs']),8)
            self.assertEqual(job['frames'],20);self.assertEqual(job['repeats'],2)
            for r in catalog['inputs']:
                split=r['new_platform_split'];trajectories[split].add(r['trajectory']);ids[split].add(r['input_id'])
        self.assertEqual([len(ids[s]) for s in ids],[24,8,8])
        for a,b in (('TRAIN','SELECT'),('TRAIN','TEST'),('SELECT','TEST')):
            self.assertFalse(trajectories[a]&trajectories[b])
        for a in p['assets'].values():
            self.assertEqual(w.ref(a['patch']['path']),a['patch'])
            g=np.load(a['patch']['path'],allow_pickle=False)
            self.assertLessEqual(abs(g.astype(int)-base.astype(int)).max(),24)
            self.assertLessEqual(np.abs(a['coefficients']).sum(),24)


if __name__=='__main__':unittest.main()
