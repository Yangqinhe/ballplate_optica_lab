"""Static replay assets, fixed native detectors and prospective analysis.

No camera, Qt, actuator or torch imports. The mean-field model and visible
eight regions are research surrogates, never drop-in old policy features.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from .contract import PROJECT,digest,validate,initial_native_matrix

STAGES=('background_start','zero_start','learned_1','zero_return_1',
        'learned_2','zero_return_2','background_end')


def pack_dmd(raw):
    a=np.asarray(raw)
    if a.shape!=(4,1,96,96) or a.dtype!=np.uint8 or not np.isin(a,(0,255)).all():
        raise ValueError('exactly four chronological binary RAW96 images required')
    mosaic=np.block([[a[0,0],a[1,0]],[a[2,0],a[3,0]]])
    return np.pad(mosaic.repeat(3,0).repeat(3,1),12)


def native_weights(matrix,shape):
    ids=np.arange(8,dtype=np.uint8).reshape(2,4).repeat(300,0).repeat(150,1)
    weights=np.stack([cv2.warpAffine((ids==k).astype(np.float32),matrix,
        (shape[1],shape[0]),flags=cv2.INTER_LINEAR) for k in range(8)])
    if ((weights<0).any() or (weights>1).any() or weights.sum(0).max()>1.00001
            or (weights.sum((1,2))<16).any()):raise ValueError('empty/overlapping fixed detector regions')
    return weights


def sums(values,weights):return np.einsum('khw,...hw->...k',weights,np.asarray(values,np.float64))


def features(stack,background,weights):
    observed=sums(stack,weights);dark=sums(background,weights)
    signal=observed-dark.mean(0)
    std=np.sqrt(signal.var(0,ddof=1)+dark.var(0,ddof=1)/len(dark))
    mean=signal.mean(0)
    return dict(mean_signed_integrals_ADU=mean.tolist(),single_frame_SD_plus_background_uncertainty_ADU=std.tolist(),
        SNR=np.divide(mean,std,out=np.zeros_like(mean),where=std>1e-12).tolist(),
        all_frame_signed_integrals_ADU=signal.tolist(),independent_pixels_assumed=False)


def score(real,predicted,weights):
    a=np.asarray(real,np.float64);b=np.asarray(predicted,np.float64)
    aa=a-a.mean();bb=b-b.mean();den=np.linalg.norm(aa)*np.linalg.norm(bb)
    actual=sums(a,weights);model=sums(b,weights)
    return dict(image_NCC=float(np.sum(aa*bb)/den) if den>1e-20 else None,
        ADU_RMSE=float(np.sqrt(np.mean((a-b)**2))),measured_native8=actual.tolist(),predicted_native8=model.tolist(),
        native8_relative_L1=float(np.abs(actual-model).sum()/max(np.abs(actual).sum(),1e-20)))


def fit_zero_gain(real,predicted):
    denominator=float(np.sum(np.square(predicted)))
    if denominator<=1e-20:raise ValueError('zero model has no signal')
    gain=float(np.sum(real*predicted)/denominator)
    if not .25<=gain<=4:raise ValueError('中心零相位固定增益超出事前范围0.25–4；不继续投学习mask')
    return gain


def analyze_case(stacks,predictions,weights,gain,gates):
    sys.path.insert(0,str(PROJECT/'tools'))
    from learned_repeat_math_v19 import zero_repeat,learned_repeat,modulation
    bg=stacks['background_start']
    dark=bg.mean(0);real=stacks['learned_1'].mean(0)-dark
    metrics=score(real,predictions['learned']*gain,weights)
    returns={name:zero_repeat(stacks['zero_start'],stacks[name],bg,gates['zero_return_gates'])
             for name in ('zero_return_1','zero_return_2')}
    repeat=learned_repeat(stacks['learned_1'],stacks['learned_2'],bg,gates['learned_repeat_gates'])
    effects={name:modulation(stacks['zero_start'],stacks[name],gates['modulation_gate'])
             for name in ('learned_1','learned_2')}
    stable=bool(repeat['passed'] and all(v['passed'] for v in returns.values())
                and all(v['passed'] for v in effects.values()))
    image_ok=metrics['image_NCC'] is not None and metrics['image_NCC']>=.75
    features_ok=metrics['native8_relative_L1']<=.2
    return dict(measured_and_prediction=metrics,zero_returns=returns,learned_repeat=repeat,
        modulation=effects,stability_passed=stable,forward_image_criterion_passed=bool(image_ok),
        native8_criterion_passed=bool(features_ok),static_criteria_passed=bool(stable and image_ok and features_ok),
        readouts=features(stacks['learned_1'],bg,weights),
        NOTE='Repeated previously seen research fixtures; not independent policy/deployment acceptance.',
        actuator_commands_generated=False,closed_loop_approved=False,release_approved=False)


class PreparedBundle:
    def __init__(self,path):
        self.path=Path(path).resolve();self.data=json.loads(self.path.read_text())
        self.catalog_sha=digest(self.path)
        d=self.data
        if d['schema']!='v20-static-research-bundle-v1' or d['release_approved'] is not False:
            raise ValueError('bounded research bundle required')
        for name,sha in d['bound_sha256'].items():
            if digest(name)!=sha:raise ValueError('bound asset/source changed: '+name)
        self.contract=validate(json.loads(Path(d['contract']).read_text()))
        self.weights=np.load(d['native_weights'],allow_pickle=False)
        shape=(self.contract['roi']['Height'],self.contract['roi']['Width'])
        if (self.weights.shape!=(8,*shape) or not np.isfinite(self.weights).all()
                or (self.weights<0).any() or (self.weights>1).any()
                or self.weights.sum(0).max()>1.00001):raise ValueError('invalid fixed native detectors')
        self.phase=np.load(d['SLM_RAW_GRAY_NO_LUT'],allow_pickle=False)
        if self.phase.shape!=(600,600) or self.phase.dtype!=np.uint8:raise ValueError('canonical SLM patch required')
        if not np.array_equal(self.phase,self.phase[::4,::4].repeat(4,0).repeat(4,1)):
            raise ValueError('original 4x4 macro-pixel structure required')
        self.cases={}
        for spec in d['cases']:
            row=dict(spec);raw=np.load(row['raw'],allow_pickle=False);dmd=np.load(row['dmd'],allow_pickle=False)
            if not np.array_equal(pack_dmd(raw),dmd):raise ValueError('DMD differs from exact four-frame packing')
            with np.load(row['prediction'],allow_pickle=False) as f:
                pred={k:f[k].copy() for k in ('zero','learned')}
            if any(a.shape!=self.weights.shape[1:] or not np.isfinite(a).all() for a in pred.values()):
                raise ValueError('native-ROI predictions invalid')
            row.update(raw_array=raw,dmd_array=dmd,predictions=pred)
            self.cases[row['id']]=row
        if set(self.cases)!={0,1,8,9,16,17}:raise ValueError('explicit six existing research fixtures required')

    def verify(self):
        if digest(self.path)!=self._sha:raise ValueError('prepared catalog changed during run')
        for name,sha in self.data['bound_sha256'].items():
            if digest(name)!=sha:raise ValueError('bound file changed during run: '+name)

    @property
    def _sha(self):return getattr(self,'catalog_sha',digest(self.path))

    def plan(self,case_ids):
        ids=list(dict.fromkeys([8,*case_ids]))
        if not ids or not set(ids)<=set(self.cases):raise ValueError('unknown test case')
        self.verify()
        return dict(schema='v20-bounded-static-run-v1',case_ids=ids,
            catalog=str(self.path),catalog_sha256=self.catalog_sha,
            frames_per_stage=8,settle_seconds=1.5,stages=list(STAGES),
            maximum_seconds=90*len(ids),max_saved_ADU_exclusive=240,incomplete_limit=0,
            gain_fit='centre8 zero_start ONLY; freeze before first learned projection, shared across ALL cases',
            gain_bounds=[.25,4.],no_geometric_refit=True,no_per_frame_or_per_region_gain=True,
            fixed_regions_from_current_zero_reference_only=True,
            original_reference_signature=self.contract['registration']['signature'],
            rules=self.data['gates'],forward_image_NCC_minimum=.75,native8_relative_L1_maximum=.2,
            no_new_SDK_connection=True,no_camera_reconfiguration=True,no_output_window_rebuild=True,
            no_auto_retry=True,no_LUT=True,hardware_pattern_association_verified=False,SLM_hardware_synchronized=False,
            fixture_replay_NOT_independent_unseen_validation=True,real_actuator_commands=False,
            old_actor_NOT_loaded=True,closed_loop_approved=False,release_approved=False)
