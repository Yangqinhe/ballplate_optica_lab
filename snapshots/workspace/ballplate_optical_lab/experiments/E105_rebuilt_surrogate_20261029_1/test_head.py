import unittest
import numpy as np
from head import solve,score,frozen_predict


class HeadTests(unittest.TestCase):
    def test_affine_budget_and_exact_background_null(self):
        rng=np.random.default_rng(6);states=rng.normal(size=(80,64))
        w=rng.normal(size=(64,2));w-=w.mean(0);y=states@w+[.2,-.1]
        x=np.repeat(states,3,axis=0);targets=np.repeat(y,3,axis=0);names=np.repeat(np.arange(80),3)
        model=solve(x,targets,names,1e-8,4,True)
        self.assertEqual(model['weights_ADU'].size+model['bias'].size,130)
        np.testing.assert_allclose(model['weights_ADU'].sum(0),0,atol=1e-10)
        p=x@model['weights_ADU']+model['bias']
        self.assertLess(np.sqrt(np.mean((p-targets)**2)),1e-4)
        np.testing.assert_allclose((x+7.4)@model['weights_ADU']+model['bias'],p,atol=1e-10)

    def test_repeat_score_is_separate_from_bias(self):
        rows=[{'input_id':'A'}]*3+[{'input_id':'B'}]*3;y=np.zeros((6,2));p=np.ones((6,2))*.1
        result=score(p,y,rows)
        self.assertAlmostEqual(result['RMS_L2_rad'],np.sqrt(.02))
        self.assertLess(result['repeat_RMS_L2_rad'],1e-15)
        with self.assertRaises(RuntimeError):solve(np.ones((3,64)),np.ones((3,2)),['A','B','C'],.1,1)

    def test_frozen_sum_head_matches_mean_units_without_clipping(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(3,64));areas=np.arange(1,65)
        w=rng.normal(size=(64,2));b=np.array([.2,-.3])
        head=dict(weights_sum=(w/areas[:,None]).tolist(),bias=b.tolist(),parameters=130)
        prediction=frozen_predict(head,x,areas)
        np.testing.assert_allclose(prediction,x@w+b,atol=1e-12)
        self.assertGreater(np.abs(prediction).max(),1)
        with self.assertRaises(RuntimeError):frozen_predict(head,x[:,:32],areas)
        with self.assertRaises(RuntimeError):frozen_predict(head,np.full((2,64),np.nan),areas)

    def test_offline_noise_sequence_covariance_and_seed(self):
        from control_budget import sequence
        cov=np.array([[.0002,.00004],[.00004,.0001]])
        config=dict(kind='measured_covariance',covariance_rad2=cov.tolist(),multiplier=2.,tau_sim_s=0.,seed=17)
        x=sequence(config,60000,1)
        np.testing.assert_allclose(np.cov(x.T),4*cov,rtol=.06,atol=1e-6)
        np.testing.assert_array_equal(x[:10],sequence(config,10,1))
        bias=sequence(dict(kind='bias',amplitude=.12,angle=np.pi/4),4,0)
        np.testing.assert_allclose(np.linalg.norm(bias,axis=1),.12,atol=1e-15)


if __name__=='__main__':unittest.main()
