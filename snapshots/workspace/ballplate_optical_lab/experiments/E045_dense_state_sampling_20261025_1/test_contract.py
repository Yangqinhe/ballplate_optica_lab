"""File-only contract/source archive fixtures; never execute launcher/UI/SDK."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SOURCE=Path(__file__).with_name('contract.py')
spec=importlib.util.spec_from_file_location('E045_contract_tested',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.plan_path=self.root/'PLAN.json';self.profile_path=self.root/'PROFILE.json'
        self.launcher=self.root/'launcher.py';self.launcher.write_text("raise AssertionError('must never execute fixture launcher')\n")
        self.extra={str(c.SELF):c.sha(c.SELF),str(c.TEST):c.sha(c.TEST)}
        self.launcher_ref=c.reference(self.launcher,c.sha(self.launcher))
        inputs={}
        for i,input_id in enumerate(c.INPUT_IDS):
            image=np.zeros((600,600),np.uint8);image[300,300+i]=255
            path=self.root/('DMD_'+str(input_id)+'.npy');np.save(path,image,allow_pickle=False)
            inputs[str(input_id)]=dict(id=input_id,dmd=c.reference(path,c.sha(path)))
        phase=self.root/'L00_canonical600.npy';np.save(phase,np.zeros((600,600),np.uint8),allow_pickle=False)
        slm=c.reference(phase,c.sha(phase))
        self.profile=dict(schema='optical-lab-current-public-profile-1',geometry_frozen=False,control_qualified=False,
            active=dict(DeviceSerialNumber='15328133',PixelFormat='Mono8',ExposureMode='Timed',ExposureAuto='Off',
                ExposureTime='74.20778274536133',GainAuto='Off',Gain='10.881464004516602',TriggerMode='On',
                TriggerSelector='FrameStart',TriggerSource='Line0',TriggerActivation='FallingEdge',AcquisitionMode='Continuous',
                OffsetX='900',OffsetY='416',Width='552',Height='554',BlackLevel='4.0771484375',opaque_nullable=None),
            actual_roi=dict(OffsetX=900,OffsetY=416,Width=552,Height=554),
            settings=dict(dmd='file',square=128,slm='file',full_panel=False,x=-410,y=4,focal_mm=400.,period=35),
            optical_reference=dict(f1_m=.15,f2_m=.1,wavelength_m=532e-9),caches=dict(slm=slm),roles=dict(fixture='preserved'))
        self.profile_path.write_text(json.dumps(self.profile));self.profile_ref=c.reference(self.profile_path,c.sha(self.profile_path))
        self.plan=dict(schema='E044-long-hold-independent-sampling-1',profile=self.profile_ref,inputs=inputs,
            source_sha256={slm['path']:slm['sha256']})
        self.plan_path.write_text(json.dumps(self.plan));self.plan_ref=c.reference(self.plan_path,c.sha(self.plan_path))
        self.patch=mock.patch.multiple(c,ROOT=self.root,PLAN_REF=self.plan_ref,PROFILE_REF=self.profile_ref)
        self.patch.start();self.addCleanup(self.patch.stop)

    def build(self):return c.build(launcher_ref=self.launcher_ref,extra_sources=self.extra)

    def test_default_import_describe_zero_files_or_modules(self):
        script="import runpy,sys;m=runpy.run_path(sys.argv[1],run_name='describe');m['main']([]);assert not any(n.startswith(('numpy','PyQt','spin_c','PySpin','mujoco')) for n in sys.modules)"
        r=subprocess.run([sys.executable,'-B','-c',script,str(SOURCE)],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(r.stdout)['default_file_reads'],0)
        with mock.patch.object(c,'pin',side_effect=AssertionError('no read')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(c.main([]),0)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):c.main(['--prepare'])

    def test_exact_seven_assets_actual_profile_and_false_geometry_preserved(self):
        value=self.build()
        self.assertEqual([r['input_id'] for r in value['allowed_dmd']],list(c.INPUT_IDS))
        self.assertEqual(value['expected_active'],self.profile['active']);self.assertIn('opaque_nullable',value['expected_active'])
        self.assertIsNone(value['expected_active']['opaque_nullable']);self.assertFalse(value['fixed_context']['geometry_frozen'])
        self.assertEqual(value['fixed_slm'],self.profile['caches']['slm']);self.assertFalse(value['control_release_approved'])
        self.assertEqual(value['maximum_frames'],240);self.assertEqual(value['minimum_post_ACK_ns'],1_500_000_000)
        self.assertEqual(value['maximum_burst_seconds'],60);self.assertEqual(value['readback_policy'],'each_frame_before_and_after')
        self.assertNotIn(str(self.root/'OUT/CONTRACT.json'),value['source_sha256'])

    def test_explicit_builder_test_launcher_and_fixed_SOURCE_SHA_cannot_be_omitted_or_overridden(self):
        for removed in (str(c.SELF),str(c.TEST)):
            extra=dict(self.extra);extra.pop(removed)
            with self.assertRaises(ValueError):c.build(launcher_ref=self.launcher_ref,extra_sources=extra)
        frozen=next(iter(c.FROZEN));extra=dict(self.extra,**{frozen:'0'*64})
        with self.assertRaises(ValueError):c.build(launcher_ref=self.launcher_ref,extra_sources=extra)
        outside=self.root/'nested';outside.mkdir();p=outside/'launcher.py';p.write_text('pass')
        with self.assertRaises(ValueError):c.build(launcher_ref=c.reference(p,c.sha(p)),extra_sources=self.extra)

    def test_full_archive_matches_original_bytes_and_exact_request_closure(self):
        result=c.prepare(self.root/'OUT',launcher_ref=self.launcher_ref,extra_sources=self.extra)
        contract=json.loads(Path(result['contract_ref']['path']).read_text())
        bundle=json.loads(Path(result['bundle']['path']).read_text())
        expected=dict(contract['source_sha256'],**{result['contract_ref']['path']:result['contract_ref']['sha256']})
        self.assertEqual(bundle['request_source_sha256'],expected)
        index=json.loads((self.root/'OUT/SOURCE_INDEX.json').read_text())
        self.assertEqual(len(index['files']),len(expected))
        for row in index['files']:
            self.assertEqual(Path(row['original']['path']).read_bytes(),Path(row['archived']['path']).read_bytes())
            self.assertEqual(c.sha(row['archived']['path']),row['original']['sha256'])
        manifest=json.loads(Path(result['manifest']['path']).read_text())
        for name,digest in manifest['file_sha256'].items():self.assertEqual(c.sha(self.root/'OUT'/name),digest)
        self.assertFalse(result['launcher_executed']);self.assertEqual(result['GUI_or_SDK_actions'],0)
        with self.assertRaises(ValueError):c.prepare(self.root/'OUT',launcher_ref=self.launcher_ref,extra_sources=self.extra)

    def test_changed_file_invalid_binary_or_fabricated_geometry_rejected(self):
        path=Path(self.plan['inputs']['0']['dmd']['path']);path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.build()
        # A new fixture provenance can still never turn false geometry true.
        self.profile['geometry_frozen']=True;self.profile_path.write_text(json.dumps(self.profile))
        new=c.reference(self.profile_path,c.sha(self.profile_path));self.plan['profile']=new
        self.plan_path.write_text(json.dumps(self.plan))
        with mock.patch.object(c,'PROFILE_REF',new),mock.patch.object(c,'PLAN_REF',c.reference(self.plan_path,c.sha(self.plan_path))):
            with self.assertRaisesRegex(ValueError,'geometry/control'):self.build()

    def test_symlink_source_and_nonnew_output_rejected_before_writes(self):
        link=self.root/'alias.py';link.symlink_to(self.launcher)
        with self.assertRaises(ValueError):c.build(launcher_ref=c.reference(link,c.sha(link)),extra_sources=self.extra)
        with self.assertRaises(ValueError):c.prepare(self.root,launcher_ref=self.launcher_ref,extra_sources=self.extra)
        self.assertFalse((self.root/'CONTRACT.json').exists())


if __name__=='__main__':unittest.main()
