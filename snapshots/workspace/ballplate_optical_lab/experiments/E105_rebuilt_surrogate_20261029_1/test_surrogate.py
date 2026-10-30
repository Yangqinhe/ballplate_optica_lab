import unittest
import numpy as np
import torch
from surrogate import ResidualMean,discrete_gray,noise_covariance,oriented,centered


class ModelTests(unittest.TestCase):
    def test_finite_response_modes_and_cancellation(self):
        from response import finite_response_modes
        j=np.eye(4);c=np.array([1.,-1.,0,0])
        r=finite_response_modes(j,c)
        np.testing.assert_allclose(r['singular_values'],np.ones(4))
        self.assertAlmostEqual(r['mixture_over_orthogonal_energy'],1.)
        j[:,1]=j[:,0];r=finite_response_modes(j,c)
        self.assertAlmostEqual(r['singular_values'][-1],0.)
        self.assertAlmostEqual(r['mixture_over_orthogonal_energy'],0.)
        q,_=np.linalg.qr(np.random.default_rng(105).normal(size=(4,4)))
        np.testing.assert_allclose(finite_response_modes(q@j,c)['singular_values'],r['singular_values'],atol=1e-12)
        with self.assertRaises(RuntimeError):finite_response_modes(np.ones((3,4)),c)
        with self.assertRaises(RuntimeError):finite_response_modes(np.full((4,4),np.nan),c)

    def test_residual_size_zero_identity_and_frozen_input_gradient(self):
        torch.set_num_threads(2);m=ResidualMean()
        self.assertEqual(sum(p.numel() for p in m.parameters()),26673)
        x=torch.randn(2,5,32,32,requires_grad=True);a=torch.randn(2,16,requires_grad=True)
        y=m(x,a);self.assertEqual(y.shape,(2,1,32,32));self.assertEqual(y.abs().sum().item(),0)
        with torch.no_grad():m.c6.weight.fill_(.01)
        for p in m.parameters():p.requires_grad_(False)
        m(x,a).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertGreater(a.grad.abs().sum().item(),0)

    def test_gray_discrete_forward(self):
        x=torch.tensor([-1.,.49,.51,128.,300.],requires_grad=True)
        np.testing.assert_array_equal(discrete_gray(x).detach().numpy(),[0,0,1,128,255])
        discrete_gray(x).sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(),[0,1,1,1,0])

    def test_noise_psd(self):
        a=np.random.default_rng(5).normal(size=(9,64));c=noise_covariance(a)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(),-1e-10)
        with self.assertRaises(ValueError):noise_covariance(np.array([[np.nan]]))

    def test_orientation_matches_numpy(self):
        a=np.arange(35).reshape(5,7)
        for code in range(8):
            expected=np.rot90(a,code%4)
            if code>=4:expected=np.fliplr(expected)
            np.testing.assert_array_equal(oriented(torch.tensor(a),code).numpy(),expected)
        padded=centered(torch.ones((4,6)),(7,9))
        self.assertEqual(padded.sum(),24);self.assertEqual(padded.shape,(7,9))

    def test_native_readout_matches_numpy_with_gradient(self):
        from residual_fit import region_means
        from dataset import integrate
        layout={'regions':[{'bounds':[0,0,4,5]},{'bounds':[3,2,7,6]}]}
        raw=torch.randn(3,6,7,requires_grad=True)
        values=region_means(raw,layout)
        expected=np.stack([integrate(x,layout)/[20,16] for x in raw.detach().numpy()])
        np.testing.assert_allclose(values.detach().numpy(),expected,rtol=1e-6,atol=1e-7)
        values.sum().backward();self.assertTrue(torch.isfinite(raw.grad).all())

    def test_development_split_separates_conditions(self):
        from residual_fit import pilot_split
        rows=[dict(input_id=i,mask=m) for i in ('CENTER','X_NEG12') for m in ('BASE','X4M')]
        self.assertEqual(pilot_split(rows),['TRAIN','SELECT_MASK','SELECT_INPUT','SELECT_BOTH'])

    def test_response_covariance_regularized_whitening(self):
        from response import covariance
        rng=np.random.default_rng(5);r=rng.normal(size=(12,64))
        c,w=covariance(r);self.assertGreater(np.linalg.eigvalsh(c).min(),0)
        np.testing.assert_allclose(w@c@w,np.eye(64),atol=1e-10)
        c,w=covariance(np.zeros((3,64)));self.assertTrue(np.isfinite(w).all())

    def test_formal_test_never_enters_selection(self):
        from residual_fit import pilot_split
        for row in ({'input_split':'TEST'}, {'input_split':'TRAIN','mask_split':'TEST'}):
            with self.assertRaises(RuntimeError):pilot_split([row])
        self.assertEqual(pilot_split([{'input_split':'TRAIN'}, {'input_split':'SELECT','mask_split':'SELECT'},
            {'input_split':'PILOT'}]),['TRAIN','SELECT_BOTH','TRAIN'])

    def test_structured_response_is_linear_and_differentiable(self):
        from residual_fit import ResponseMean
        torch.set_num_threads(2);net=ResponseMean();x=torch.randn(1,5,32,32)
        a=torch.zeros(1,16);a[0,0]=3;a.requires_grad_(True)
        with torch.no_grad():net.c6.weight.normal_(0,.01)
        p=net(x,a);m=net(x,-a);b=net(x,torch.zeros_like(a))
        torch.testing.assert_close((p+m)/2,b,rtol=1e-5,atol=1e-7)
        p.sum().backward();self.assertGreater(abs(a.grad[0,0]),0)
        self.assertLess(sum(p.numel() for p in net.parameters()),30000)

    def test_signed_pairs_no_input_or_direction_leak(self):
        from residual_fit import signed_pairs
        rows=[dict(input_id=i,coefficients_raw_gray=c) for i,c in
              [('A',[6,6,-6,-6]),('A',[-6,-6,6,6]),('B',[6,6,-6,-6]),('A',[0,0,0,0])]]
        self.assertEqual(signed_pairs(rows),[(0,1)])
        with self.assertRaises(RuntimeError):signed_pairs(rows+[rows[0]])

    def test_quadratic_residual_identifiable_diagonals_only(self):
        from residual_fit import ResponseMean
        net=ResponseMean(quadratic=True);x=torch.zeros(1,5,8,8)
        with torch.no_grad():
            net.c6.bias.copy_(torch.tensor([1.,2.,0.,0.,0.,3.,0.,0.,0.]))
        a=torch.zeros(1,16);a[0,0]=3
        p=net(x,a);m=net(x,-a);b=net(x,torch.zeros_like(a))
        torch.testing.assert_close(p,torch.full_like(p,6.))
        torch.testing.assert_close(m,torch.full_like(m,2.))
        torch.testing.assert_close(b,torch.ones_like(b))
        torch.testing.assert_close((p-m)/2,torch.full_like(p,2.))
        self.assertEqual(sum(p.numel() for p in net.parameters()),25177)

    def test_frozen_proxy_gray_bounds_and_gradient(self):
        from residual_fit import FrozenProxy
        proxy=object.__new__(FrozenProxy);proxy.base=torch.zeros(75,75,dtype=torch.float64)
        proxy.bases=torch.ones(4,75,75,dtype=torch.float64);proxy.max_l1=24
        c=torch.tensor([3.,-2.,1.,0.],dtype=torch.float64,requires_grad=True)
        g=proxy.gray(c);self.assertEqual(g.shape,(75,75))
        g.mean().backward();torch.testing.assert_close(c.grad,torch.ones(4,dtype=torch.float64),atol=1e-12,rtol=1e-12)
        with self.assertRaises(RuntimeError):proxy.gray(torch.tensor([25.,0.,0.,0.]))
        with self.assertRaises(RuntimeError):proxy.gray(torch.tensor([float('nan'),0.,0.,0.]))
        with self.assertRaises(RuntimeError):proxy.gray(torch.zeros(16))

    def test_context_and_cross_residual_fixed_budget_and_products(self):
        from residual_fit import make_network
        torch.set_num_threads(2)
        for name in ('response_context','response_cross_context'):
            net=make_network(name);x=torch.randn(1,5,32,32);a=torch.zeros(1,16)
            self.assertLess(sum(p.numel() for p in net.parameters()),30000)
            self.assertEqual(net(x,a).abs().sum().item(),0)
        net=make_network('response_cross_context')
        with torch.no_grad():net.c6.bias[9]=2
        a[0,0]=3;a[0,1]=-1.5;a.requires_grad_(True)
        y=net(x,a);torch.testing.assert_close(y,torch.full_like(y,-1.))
        y.mean().backward();self.assertNotEqual(a.grad[0,0].item(),0)

    def test_forecast_commitment_rejects_old_frames_changed_masks_and_predictions(self):
        import json
        import tempfile
        from pathlib import Path
        from workflow import write,ref
        from validate_physics import forecast_evidence_check
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);forecast=root/'FORECAST';cache=root/'CACHE';evaluation=root/'EVAL'
            # Deliberately mutate disposable fixtures to exercise rejection;
            # experiment outputs continue to use exclusive-create write().
            def fixture(path,value):path.write_text(json.dumps(value),encoding='utf-8')
            for p in (forecast,cache,evaluation):p.mkdir()
            rows=[dict(input_id=str(i),dmd_reference=dict(sha256='d'+str(i)),mask_reference=dict(sha256='m')) for i in range(80)]
            image=np.zeros((80,2,2),np.float32)
            np.savez(forecast/'PREDICTIONS.npz',predicted=image,physics=image)
            np.savez(evaluation/'PREDICTIONS.npz',predicted=image,physics=image)
            write(root/'DATA.json',dict(rows=[dict(host_times_ns=[11])]))
            index=dict(rows=rows,dataset=ref(root/'DATA.json'),physical_model='fixed',layout='fixed')
            write(cache/'INDEX.json',index)
            write(forecast/'FORECAST.json',dict(rows=rows,model='fixed',physical_model='fixed',layout='fixed',
                predictions=ref(forecast/'PREDICTIONS.npz'),created_unix_ns=10))
            def evaluation_result():fixture(evaluation/'RESULT.json',dict(formal_TEST=True,model='fixed',cache=ref(cache/'INDEX.json')))
            evaluation_result();forecast_evidence_check(forecast,cache,evaluation,root/'PASS')
            image[3,0,0]=1
            np.savez(evaluation/'PREDICTIONS.npz',predicted=image,physics=np.zeros_like(image))
            with self.assertRaises(RuntimeError):forecast_evidence_check(forecast,cache,evaluation,root/'BAD_PRED')
            np.savez(evaluation/'PREDICTIONS.npz',predicted=np.zeros_like(image),physics=np.zeros_like(image))
            write(root/'DATA_OLD.json',dict(rows=[dict(host_times_ns=[9])]))
            index['dataset']=ref(root/'DATA_OLD.json');fixture(cache/'INDEX.json',index);evaluation_result()
            with self.assertRaises(RuntimeError):forecast_evidence_check(forecast,cache,evaluation,root/'BAD_TIME')
            index['dataset']=ref(root/'DATA.json');index['rows'][0]['dmd_reference']['sha256']='changed'
            fixture(cache/'INDEX.json',index);evaluation_result()
            with self.assertRaises(RuntimeError):forecast_evidence_check(forecast,cache,evaluation,root/'BAD_DMD')

    def test_readout_surrogate_uses_pixels_and_psd_local_kernel(self):
        from readout_surrogate import image_features,kernel
        a=np.zeros((600,600),np.uint8)
        for y in (0,300):
            for x in (0,300):a[y+140:y+147,x+142:x+149]=255
        f=image_features(a);g=image_features(np.roll(a,2,axis=1))
        self.assertEqual(f.shape,(24,));self.assertTrue(np.isfinite(f).all())
        # Relative historical positions stay fixed; final image center shifts.
        np.testing.assert_allclose(g[:18],f[:18],atol=1e-12)
        self.assertAlmostEqual(g[18]-f[18],2/300)
        with self.assertRaises(RuntimeError):image_features(np.zeros_like(a))
        rng=np.random.default_rng(5);x=rng.normal(size=(12,24));c=rng.normal(size=(12,4));c=c/np.abs(c).sum(1)[:,None]
        for degree in (1,2):
            k=kernel(x,c,x,c,1.,degree)
            np.testing.assert_allclose(k,k.T,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-10)

    def test_focus_fft_residual_is_linear_convolution_not_wrapped_correlation(self):
        from convolution_residual import FocusConvolution
        torch.set_num_threads(2);model=FocusConvolution(5)
        source=torch.randn(2,13,17);c=torch.randn(2,4)/8;shape=(18,24)
        spectra=torch.fft.rfft2(source,s=shape)
        self.assertEqual(model(spectra,c,(13,17),shape).abs().sum().item(),0)
        with torch.no_grad():model.kernels.normal_(0,.1)
        cc=torch.cat((torch.ones(2,1),c),1);k=torch.einsum('bk,khw->bhw',cc,model.kernels)/25
        expected=torch.cat([torch.nn.functional.conv2d(source[i][None,None],k[i].flip(-2,-1)[None,None],padding=2)[0] for i in range(2)])
        actual=model(spectra,c,(13,17),shape)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        c.requires_grad_(True);model(spectra,c,(13,17),shape).square().mean().backward()
        self.assertGreater(c.grad.abs().sum().item(),0)

    def test_coarse_convolution_keeps_wide_support_with_small_parameter_count(self):
        from convolution_residual import FocusConvolution
        import torch.nn.functional as F
        torch.set_num_threads(2);model=FocusConvolution(33,5)
        self.assertEqual(sum(p.numel() for p in model.parameters()),125)
        with torch.no_grad():model.kernels.normal_(0,.1)
        source=torch.randn(1,13,17);c=torch.randn(1,4,requires_grad=True);shape=(48,50)
        actual=model(torch.fft.rfft2(source,s=shape),c,(13,17),shape)
        kernels=F.interpolate(model.kernels[:,None],(33,33),mode='bilinear',align_corners=False)[:,0]
        kernel=torch.einsum('bk,khw->bhw',torch.cat((torch.ones(1,1),c),1),kernels)/33**2
        expected=F.conv2d(source[:,None],kernel.flip((-2,-1))[:,None],padding=16)[:,0]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        actual.sum().backward();self.assertGreater(c.grad.abs().sum(),0)

    def test_static_baseline_uses_only_train_conditions(self):
        import tempfile
        from pathlib import Path
        from workflow import write,ref,read
        from research_report import freeze_static
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);forecast=root/'F';forecast.mkdir()
            physical=np.full((3,8,8),10.,np.float32)
            measured=physical+np.array([1.,3.,100.],np.float32)[:,None,None]
            np.savez(root/'TRAIN.npz',truth=measured,physics=physical)
            write(root/'MODEL.json',dict(predictions=ref(root/'TRAIN.npz')))
            write(root/'PLAN.json',dict(train=[0,1]))
            write(forecast/'FORECAST.json',dict(plan=ref(root/'PLAN.json'),model=ref(root/'MODEL.json')))
            freeze_static(forecast,root/'STATIC')
            np.testing.assert_array_equal(np.load(root/'STATIC/STATIC_CORRECTION.npy'),np.full((8,8),2.))
            self.assertTrue(read(root/'STATIC/FREEZE.json')['no_TEST_read'])

    def test_sealed_score_uses_only_committed_predictions_and_rejects_changed_bytes(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from workflow import write,ref
        from validate_physics import evaluate_sealed_predictions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);forecast=root/'F';data=root/'D';forecast.mkdir();data.mkdir()
            layout=dict(regions=[dict(bounds=[x,y,x+1,y+1],pixels=1) for y in range(8) for x in range(8)])
            rng=np.random.default_rng(5);means=np.stack([np.ones(64)*v for v in (10.1,10.2,10.05,10.3,10.2,10.4)])
            np.savez(root/'features.npz',mean_sums=means+rng.normal(0,.001,means.shape))
            write(root/'TRAIN_DATA.json',dict(baseline='fixed',features=ref(root/'features.npz')))
            write(root/'CACHE.json',dict(dataset=ref(root/'TRAIN_DATA.json'),rows=[dict(visits=[0,1,2]),dict(visits=[3,4,5])]))
            write(root/'PLAN.json',dict(cache=ref(root/'CACHE.json'),train=[0,1],state_scales_ADU=[1.]*64))
            write(root/'MODEL.json',dict(frozen=True))
            truth=np.stack([np.full((8,8),11.,np.float32),np.full((8,8),13.,np.float32)])
            np.savez(forecast/'PREDICTIONS.npz',predicted=truth,physics=np.full_like(truth,10))
            rows=[dict(input_id='A',dmd_reference=dict(sha256='d'),mask_reference=dict(sha256=s)) for s in ('p','m')]
            write(forecast/'FORECAST.json',dict(rows=rows,model=ref(root/'MODEL.json'),plan=ref(root/'PLAN.json'),
                layout=layout,predictions=ref(forecast/'PREDICTIONS.npz'),created_unix_ns=10,known_inputs_only=True))
            write(data/'INDEX.json',dict(layout=layout,baseline='fixed',rows=[dict(host_times_ns=[11])]))
            samples=[dict(input_id='A',label=dict(dmd=dict(sha256='d')),mask=dict(sha256=s),mask_name=s,
                coefficients=[v,0,0,0],image=truth[i],visits=[i*3+j for j in range(3)]) for i,(s,v) in enumerate((('p',1),('m',-1)))]
            with patch('fit.groups',return_value=samples):evaluate_sealed_predictions(forecast,data,root/'PASS')
            result=json.loads((root/'PASS/RESULT.json').read_text());self.assertTrue(result['scored_exact_PRE_capture_arrays'])
            self.assertEqual(result['metrics'][0]['predicted']['region_RMS_ADU'],0)
            samples[0]['mask']['sha256']='wrong'
            with patch('fit.groups',return_value=samples):
                with self.assertRaises(RuntimeError):evaluate_sealed_predictions(forecast,data,root/'BAD_SHA')
if __name__=='__main__':unittest.main()
