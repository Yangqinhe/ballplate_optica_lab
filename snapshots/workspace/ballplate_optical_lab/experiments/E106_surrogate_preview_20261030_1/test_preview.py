"""Contract/math tests only; no GUI, camera, USB, projection or model training."""
from copy import deepcopy
import sys
import unittest
import numpy as np
import model


def fixture():
    m=model.read(model.ROOT/'MODEL_01/MODEL.json');e=m['expected']
    c=dict(camera=deepcopy(e['camera_actual']),roi=deepcopy(e['roi']),settings=deepcopy(e['slm']),
        optics=deepcopy(e['optical_reference']),geometry=deepcopy(m['geometry']),
        orientation=e['orientation']['code'],dmd_side=600)
    b=model.read(model.checked(m['baseline']))
    return m,c,np.load(model.checked(b['files']['dmd'])),np.load(model.checked(b['files']['slm']))


class PreviewTests(unittest.TestCase):
    def setUp(self):self.m,self.c,self.dmd,self.mask=fixture()
    def test_no_hardware_or_torch_on_import(self):
        self.assertNotIn('PySpin',sys.modules);self.assertNotIn('torch',sys.modules)
        self.assertNotIn('PyQt5',sys.modules)
    def test_original_mask_and_context(self):
        self.assertEqual(model.validate_request(self.dmd,self.mask,self.c,self.m)['name'],'BASE')
    def test_changed_exposure_gain_black_rejected_without_hardware_action(self):
        for field in ('ExposureTime','Gain','BlackLevel'):
            with self.subTest(field=field):
                c=deepcopy(self.c);c['camera'][field]=str(float(c['camera'][field])+1)
                with self.assertRaises(ValueError):model.validate_context(c,self.m)
    def test_changed_ROI_rejected(self):
        self.c['roi']['OffsetX']+=1
        with self.assertRaises(ValueError):model.validate_context(self.c,self.m)
    def test_changed_scale_rejected(self):
        self.c['dmd_side']=1080
        with self.assertRaises(ValueError):model.validate_context(self.c,self.m)
    def test_changed_SLM_geometry_rejected(self):
        for k,v in (('x',33),('y',107),('full_panel',True),('slm','zero')):
            with self.subTest(key=k):
                c=deepcopy(self.c);c['settings'][k]=v
                with self.assertRaises(ValueError):model.validate_context(c,self.m)
    def test_changed_mapping_or_orientation_rejected(self):
        c=deepcopy(self.c);c['geometry']['matrix_sensor'][0][2]+=.1
        with self.assertRaises(ValueError):model.validate_context(c,self.m)
        self.c['orientation']=0
        with self.assertRaises(ValueError):model.validate_context(self.c,self.m)
    def test_unknown_mask_not_silently_projected_to_base(self):
        self.mask[0,0]^=1
        with self.assertRaises(ValueError):model.validate_request(self.dmd,self.mask,self.c,self.m)
    def test_binary_DMD_required(self):
        self.dmd[0,0]=128
        with self.assertRaises(ValueError):model.validate_request(self.dmd,self.mask,self.c,self.m)
    def test_missing_camera_or_geometry_rejected(self):
        for field in ('camera','geometry','optics'):
            with self.subTest(field=field):
                c=deepcopy(self.c);c.pop(field)
                with self.assertRaises(ValueError):model.validate_context(c,self.m)
    def test_identity_changes_with_pixels_or_context(self):
        key=model.request_key(self.dmd,self.mask,self.c)
        a=self.dmd.copy();a[0,0]^=255
        self.assertNotEqual(key,model.request_key(a,self.mask,self.c))
        c=deepcopy(self.c);c['dmd_side']=1080
        self.assertNotEqual(key,model.request_key(self.dmd,self.mask,c))
        self.assertEqual(key,model.request_key(self.dmd.copy(),self.mask.copy(),deepcopy(self.c)))
    def test_preview_does_not_mutate_arrays_or_context(self):
        d=self.dmd.copy();m=self.mask.copy();c=deepcopy(self.c)
        model.validate_request(self.dmd,self.mask,self.c,self.m)
        np.testing.assert_array_equal(d,self.dmd);np.testing.assert_array_equal(m,self.mask)
        self.assertEqual(c,self.c)
    def test_all_registered_masks_within_local_bounds(self):
        for item in self.m['masks'].values():
            a=np.load(model.checked(item['source']))
            self.assertEqual(model.validate_request(self.dmd,a,self.c,self.m),item)
            self.assertLessEqual(sum(abs(x) for x in item['coefficients']),24)


if __name__=='__main__':unittest.main()
