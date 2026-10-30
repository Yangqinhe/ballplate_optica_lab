import unittest
import numpy as np
from optical_lab.metrics import compare_model, describe_real, paired_response, ncc


def fixture():
    labels = np.arange(64).reshape(8, 8) % 8
    regions = np.stack([labels == i for i in range(8)])
    mean = np.arange(64).reshape(8, 8)/4+10
    return np.stack([mean-1, mean, mean+1]), regions


class MetricsTests(unittest.TestCase):
    def test_exact_model(self):
        a,r=fixture(); m=compare_model(a,a.mean(0),r)
        self.assertEqual(m['raw_RMSE_ADU'],0)
        self.assertAlmostEqual(m['NCC'],1)
        self.assertAlmostEqual(m['native8_relative_L2'],0)

    def test_gain_not_silently_fitted(self):
        a,r=fixture(); m=compare_model(a,a.mean(0)*2,r)
        self.assertGreater(m['raw_RMSE_ADU'],10)
        self.assertAlmostEqual(m['NCC'],1)

    def test_noise_and_efficiency(self):
        a,r=fixture(); m=describe_real(a,np.ones_like(a)*10,r)
        self.assertAlmostEqual(m['single_frame_repeat_pixel_RMS_ADU'],1)
        self.assertIsNone(m['absolute_optical_efficiency'])

    def test_bad_regions_and_shape(self):
        a,r=fixture()
        with self.assertRaises(ValueError): compare_model(a,np.zeros((4,4)),r)
        with self.assertRaises(ValueError): describe_real(a,regions=np.ones_like(r))
        with self.assertRaises(ValueError): describe_real(a[:1])

    def test_no_contrast_unknown(self):
        self.assertIsNone(ncc(np.ones((4,4)),np.ones((4,4))))
        self.assertIsNone(compare_model(np.ones((2,4,4)),np.ones((4,4)))['RMSE_over_real_spatial_contrast'])

    def test_head_is_raw_and_not_control(self):
        a,r=fixture(); head={'weight_raw_ADU_to_rad':np.ones((2,8))*.001,'bias_rad':[0,0]}
        m=compare_model(a,a.mean(0)+1,r,head)
        self.assertAlmostEqual(m['raw_head_mean_action_MAE_rad'],.064)
        self.assertTrue(m['head_diagnostic_NOT_a_closed_loop'])

    def test_response_direction(self):
        a,r=fixture(); b=a+2
        m=paired_response(a,b,r,a.mean(0),b.mean(0))
        self.assertAlmostEqual(m['model_to_real_response_cosine'],1)
        self.assertAlmostEqual(m['response_relative_L2_error'],0)
        self.assertFalse(m['statistical_significance_or_absolute_efficiency'])

    def test_zero_response_unknown(self):
        a,r=fixture(); m=paired_response(a,a,r,a.mean(0),a.mean(0))
        self.assertIsNone(m['model_to_real_response_cosine'])


if __name__ == '__main__': unittest.main()
