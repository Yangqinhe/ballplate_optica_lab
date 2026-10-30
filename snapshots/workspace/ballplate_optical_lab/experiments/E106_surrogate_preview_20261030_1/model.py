"""Read-only, frozen E105 mean prediction. No camera/display/Qt on import.

Heavy inference lives in a separate process, never the camera or GUI thread.
This is a visualization model, not an online image-processing/control stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
EXPERIMENT = LAB/'experiments/E105_rebuilt_surrogate_20261029_1'
MODEL = EXPERIMENT/'AXES_RESPONSE_105'
STATE_SHA = 'd59edecf69df759372f70bdbb76110b050ba332ad03b1bad93734a9416d1feef'
RESULT_SHA = '32eae6872640cf12778848a279447f6d7e8e2da46c0d3b50fc724c6e08e8bd3f'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=digest(path))
def read(path): return json.loads(Path(path).read_text())
def pixels(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checked(item):
    if digest(item['path']) != item['sha256']: raise ValueError('来源SHA不一致：'+item['path'])
    return Path(item['path'])
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(folder):
    """Explicit fixed model; no latest/mtime lookup and no hardware imports."""
    if folder.exists(): raise ValueError('fresh manifest directory required')
    if digest(MODEL/'STATE.pt') != STATE_SHA or digest(MODEL/'RESULT.json') != RESULT_SHA:
        raise ValueError('E105 frozen winner changed')
    baseline=read(EXPERIMENT/'BASELINE_01/BASELINE.json')
    plan=read(EXPERIMENT/'DESIGN_STRONG_N3_01/PLAN.json')
    masks={}
    for name, asset in plan['assets'].items():
        image=np.load(checked(asset['patch']), allow_pickle=False)
        if image.shape!=(600,600) or image.dtype!=np.uint8: raise ValueError('native SLM patch required')
        masks[pixels(image)]=dict(name=name, coefficients=asset['coefficients'], source=asset['patch'])
    cache=read(checked(read(MODEL/'PLAN.json')['cache']))
    known={}
    for row in cache['rows']:
        item=row['dmd_reference']
        if item['path'] not in known:
            a=np.load(checked(item),allow_pickle=False)
            known[item['path']]=dict(pixel_sha256=pixels(a),input_id=row['input_id'])
    known_pixels={r['pixel_sha256']:r['input_id'] for r in known.values()}
    folder.mkdir()
    result=dict(schema='E106-preview-1',model=ref(MODEL/'RESULT.json'),state=ref(MODEL/'STATE.pt'),
        plan=ref(MODEL/'PLAN.json'),engine=ref(EXPERIMENT/'residual_fit.py'),
        baseline=ref(EXPERIMENT/'BASELINE_01/BASELINE.json'),expected=baseline['expected'],
        geometry=baseline['snapshot']['geometry'],masks=masks,known_inputs=known_pixels,
        source=ref(__file__),prediction='conditional mean in native RAW ADU; no noise samples',
        original_mask_not_retrained=True,deployment_qualified=False)
    write(folder/'MODEL.json',result)
    return ref(folder/'MODEL.json')


def validate_context(context, manifest):
    expected=manifest['expected'];camera=context.get('camera',{})
    for key in ('DeviceSerialNumber','PixelFormat','ExposureMode','ExposureAuto','GainAuto',
                'TriggerMode','TriggerSource','TriggerActivation','SensorShutterMode'):
        if camera.get(key)!=expected['camera_actual'][key]: raise ValueError('相机配置失配：'+key)
    for key in ('ExposureTime','Gain','BlackLevel'):
        try: value=float(camera[key]);wanted=float(expected['camera_actual'][key])
        except (KeyError,TypeError,ValueError): raise ValueError('相机实际值缺失：'+key)
        if not np.isfinite(value) or not np.isclose(value,wanted,rtol=1e-7,atol=1e-6):
            raise ValueError('模型光度配置失配：'+key)
    if context.get('roi')!=expected['roi']: raise ValueError('ROI与模型不一致')
    if context.get('dmd_side')!=600: raise ValueError('模型只覆盖DMD 1.00×；不缩放旧预测冒充新结果')
    s=context.get('settings',{})
    for key in ('x','y','full_panel'):
        if s.get(key)!=expected['slm'][key]: raise ValueError('SLM位置/有效窗口已改变：'+key)
    if s.get('slm')!='file': raise ValueError('请选择已验证的SLM载入图；零相位/透镜不在本模型范围')
    for key,wanted in expected['optical_reference'].items():
        if context.get('optics',{}).get(key)!=wanted: raise ValueError('仿真参数已改变：'+key)
    geo=context.get('geometry')
    if not isinstance(geo,dict): raise ValueError('需要当前已锁定的倍率参考')
    original=manifest['geometry']
    if geo.get('orientation')!=original['orientation'] or context.get('orientation')!=original['orientation']:
        raise ValueError('CCD方向与模型不一致')
    matrix=np.asarray(geo.get('matrix_sensor',[]))
    if matrix.shape!=(2,3) or not np.array_equal(matrix,np.asarray(original['matrix_sensor'])):
        raise ValueError('倍率/位置参考已改变；需要重新标定代理模型')
    if geo.get('dmd_side',600)!=600 or not geo.get('frozen'): raise ValueError('倍率参考未锁定或尺度不符')


def validate_request(dmd, mask, context, manifest):
    validate_context(context,manifest)
    if dmd.dtype!=np.uint8 or dmd.shape!=(600,600) or not np.isin(dmd,(0,255)).all():
        raise ValueError('需要600×600二值DMD请求图')
    if mask.dtype!=np.uint8 or mask.shape!=(600,600): raise ValueError('需要600×600原始SLM灰度图')
    item=manifest['masks'].get(pixels(mask))
    if item is None: raise ValueError('当前SLM不在冻结模型的已知局部探针清单；不使用旧预测')
    return item


def request_key(dmd,mask,context):
    raw=json.dumps(context,sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(raw+pixels(dmd).encode()+pixels(mask).encode()).hexdigest()


class Engine:
    def __init__(self,manifest,device=None):
        self.manifest=manifest
        for key in ('model','state','plan','engine','baseline'): checked(manifest[key])
        if manifest['state']['sha256']!=STATE_SHA or manifest['model']['sha256']!=RESULT_SHA:
            raise ValueError('unexpected frozen model')
        sys.path.insert(0,str(EXPERIMENT))
        import torch
        from residual_fit import FrozenProxy
        torch.set_num_threads(2)
        self.torch=torch;self.proxy=FrozenProxy(MODEL,device=device)

    def predict(self,dmd,mask,context):
        item=validate_request(dmd,mask,context,self.manifest);start=time.monotonic()
        p=self.proxy;t=self.torch;key=request_key(dmd,mask,context)
        # One current source: no growth with arbitrary manual input changes.
        p.relay.sources.clear();p.base_features.clear();p.source_keys.clear()
        with t.inference_mode():
            c=t.tensor(item['coefficients'],device=p.device,dtype=t.float32)
            actual=t.round(p.gray(c).clamp(0,255)).repeat_interleave(8,0).repeat_interleave(8,1)
            if not np.array_equal(actual.cpu().numpy().astype(np.uint8),mask):
                raise ValueError('实际mask与冻结系数对应的量化图不一致')
            corrected=p(key,dmd,c).cpu().numpy()
            physical=p.relay(key,dmd,p.gray(c))[0].cpu().numpy()
        return dict(corrected=corrected,physical=physical),dict(key=key,mask=item['name'],
            input_id=self.manifest['known_inputs'].get(pixels(dmd)),
            unseen_input=pixels(dmd) not in self.manifest['known_inputs'],
            seconds=time.monotonic()-start,model_state_sha256=STATE_SHA,
            stochastic_noise_added=False,control_or_capture_modified=False)


def serve(manifest_path):
    """Serialized stdin requests; stdout is only small tagged JSON messages."""
    manifest=read(manifest_path);engine=Engine(manifest)
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line);archive=Path(request['archive']).resolve()
            with np.load(archive,allow_pickle=False) as a:
                images,meta=engine.predict(a['dmd'],a['mask'],request['context'])
            output=archive.parent/'result.npz';temporary=archive.parent/'result.tmp'
            with temporary.open('wb') as f:np.savez(f,**images)
            os.replace(temporary,output)
            response=dict(ok=True,token=request['token'],archive=str(output),metadata=meta)
        except Exception as error:
            response=dict(ok=False,token=request.get('token'),error=str(error))
        print('E106 '+json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    g.add_argument('--prepare',type=Path);g.add_argument('--serve',type=Path)
    a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.prepare),ensure_ascii=False))
    elif a.serve:serve(a.serve)
    else:print('{"hardware_actions":0,"inference_started":false}')
