"""Deterministic offline arithmetic tests; no model, SDK or GUI imports."""
from copy import deepcopy
import subprocess
import sys
import unittest

import numpy as np

from optical_lab.readout import CONFIG_REQUESTED, integrate_pixels, linear_gain_head, preprocess_intensity, validate_config


class ReadoutTests(unittest.TestCase):
    def test_requested_formula_no_raw_or_config_mutation(self):
        raw=np.array([[0,10,11,12,37,38,255]],dtype=np.uint8);before=raw.copy();config=deepcopy(CONFIG_REQUESTED)
        out=preprocess_intensity(raw,config)
        np.testing.assert_array_equal(raw,before);self.assertEqual(config,CONFIG_REQUESTED)
        np.testing.assert_array_equal(out,[[0,0,4.75,14.25,251.75,255,255]])
        self.assertEqual(out.dtype,np.float64);self.assertFalse(np.shares_memory(out,raw))

    def test_exact_threshold_and_software_saturation_boundary(self):
        top=10.5+255/9.5;self.assertAlmostEqual(top,37.3421052631579)
        raw=np.array([[10.5,top-1e-8,top,top+1e-8]])
        out=preprocess_intensity(raw)
        self.assertEqual(out[0,0],0);self.assertLess(out[0,1],255)
        self.assertAlmostEqual(out[0,2],255,places=12);self.assertEqual(out[0,3],255)

    def test_display_output_rounds_but_float_does_not(self):
        raw=np.array([[10.6,11,12]],float)
        f=preprocess_intensity(raw);c=dict(CONFIG_REQUESTED,output="display_uint8");u=preprocess_intensity(raw,c)
        np.testing.assert_array_equal(u,np.rint(f).astype(np.uint8));self.assertEqual(f.dtype,np.float64)
        self.assertTrue(np.any(f!=u));self.assertEqual(u.dtype,np.uint8)
        c=dict(CONFIG_REQUESTED,gamma=.5);np.testing.assert_allclose(preprocess_intensity(raw,c),255*np.sqrt(np.clip((raw-10.5)*9.5/255,0,1)))

    def test_noisy_sample_and_model_mean_are_distinct_domains(self):
        # Passing an expected ADU mean is supported, but thresholding it does
        # not reproduce the expectation of thresholded noisy single frames.
        samples=np.array([[[10.]],[[11.]]]);mean=samples.mean(0)
        self.assertNotEqual(float(preprocess_intensity(samples).mean()),float(preprocess_intensity(mean)[0,0]))
        self.assertEqual(preprocess_intensity(samples).shape,samples.shape)

    def test_exact_config_keys_and_invalid_numbers_fail_closed(self):
        changes=[dict(gain=True),dict(gain=np.bool_(False)),dict(gain=.5),dict(gain=129),dict(gain=np.inf),
            dict(black_ADU=-1),dict(black_ADU=256),dict(black_ADU=np.nan),dict(gamma=.29),dict(gamma=3.01),
            dict(gamma=False),dict(output="raw"),dict(mode="display"),dict(schema="other"),dict(extra=1)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(dict(CONFIG_REQUESTED,**change))
        missing=deepcopy(CONFIG_REQUESTED);missing.pop("gain")
        with self.assertRaises(ValueError):validate_config(missing)

    def test_invalid_images_rejected_without_fallback(self):
        for value in (np.array([[True]]),np.array([[-1.]]),np.array([[np.nan]]),np.array([[np.inf]]),
                      np.ones((1,1),dtype=np.uint16),np.ones(8),np.ones((1,1,1,1)),np.empty((0,2))):
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):preprocess_intensity(value)

    def test_integrals_are_per_frame_float64_no_temporal_average(self):
        weights=np.zeros((8,2,4));weights.reshape(8,-1)[np.arange(8),np.arange(8)]=1
        raw=np.stack((np.arange(8,dtype=np.uint8).reshape(2,4),np.full((2,4),20,np.uint8)))
        before=raw.copy();wb=weights.copy();out=integrate_pixels(raw,weights)
        np.testing.assert_array_equal(out,[np.arange(8),np.full(8,20)]);self.assertEqual(out.dtype,np.float64)
        np.testing.assert_array_equal(raw,before);np.testing.assert_array_equal(weights,wb)
        np.testing.assert_array_equal(integrate_pixels(raw[0],weights),np.arange(8))
        weights[0,0,0]=.5;self.assertEqual(integrate_pixels(raw[1],weights)[0],10)

    def test_weight_shape_overlap_bounds_and_nonfinite_rejected(self):
        x=np.ones((2,4));w=np.zeros((8,2,4))
        bad=[np.zeros((7,2,4)),np.zeros((8,4,2))]
        for v in (-1,1.1,np.nan,np.inf):
            a=w.copy();a[0,0,0]=v;bad.append(a)
        a=w.copy();a[0,0,0]=.4;a[1,0,0]=.4;bad.append(a)
        for a in bad:
            with self.assertRaises(ValueError):integrate_pixels(x,a)

    def test_pure_gain_fold_preserves_actions_and_deepcopies_head(self):
        rng=np.random.default_rng(817);w=rng.normal(size=(2,8));b=rng.normal(size=2)
        head=dict(schema="historical",weight_raw_ADU_to_rad=w.tolist(),bias_rad=b.tolist(),nested=dict(value=[1]),short_MuJoCo_pilot_eligible=True)
        before=deepcopy(head);new=linear_gain_head(head,9.5);f=rng.uniform(1,100,size=(3,8))
        np.testing.assert_allclose(f@w.T+b,(f*9.5)@np.array(new["weight_raw_ADU_to_rad"]).T+new["bias_rad"],rtol=1e-14,atol=1e-13)
        self.assertEqual(head,before);new["nested"]["value"].append(2);self.assertEqual(head,before)
        self.assertFalse(new["algebraic_transform"]["supports_requested_clip"] or new["short_MuJoCo_pilot_eligible"])
        self.assertFalse(new["release_approved"])

    def test_pure_gain_fold_does_not_claim_threshold_equivalence(self):
        w=np.ones((2,8));head=dict(weight_raw_ADU_to_rad=w.tolist(),bias_rad=[0,0]);new=linear_gain_head(head,9.5)
        raw=np.full((2,4),10,np.uint8);regions=np.eye(8).reshape(8,2,4)
        old=integrate_pixels(raw,regions)@w.T
        clipped=integrate_pixels(preprocess_intensity(raw),regions)@np.array(new["weight_raw_ADU_to_rad"]).T
        self.assertFalse(np.array_equal(old,clipped))
        for value in (True,0,129,np.nan):
            with self.assertRaises(ValueError):linear_gain_head(head,value)
        with self.assertRaises(ValueError):linear_gain_head(dict(weight_raw_ADU_to_rad=[[1]],bias_rad=[1]),2)

    def test_clean_import_has_no_torch_or_hardware(self):
        subprocess.run([sys.executable,"-c","import sys;from optical_lab import readout;assert not any(k.startswith(('torch','PySpin','PyQt','gi'))for k in sys.modules)"],check=True)


if __name__=="__main__":unittest.main()
