"""DMD-video -> native-CCD point calibration. Pure math; no hardware imports."""
from dataclasses import asdict, dataclass
import hashlib
import json

import cv2
import numpy as np


@dataclass(frozen=True)
class SpatialPlan:
    mode: str = 'pilot'
    radius: int = 8
    frames: int = 3
    discard: int = 2
    settle_s: float = .7
    timeout_s: float = 240.
    min_signal_ADU: float = 3.
    noise_sigma: float = 5.
    min_area: int = 9
    edge_margin: int = 5
    validation_p95_px: float = 1.
    validation_max_px: float = 2.
    return_drift_px: float = 1.

    def validate(self):
        if self.mode not in ('pilot', 'grid'):
            raise ValueError('unknown spatial protocol')
        for name, low, high in (('radius',4,24),('frames',3,5),('discard',2,4),
                               ('min_area',9,100),('edge_margin',5,30)):
            value=getattr(self,name)
            if type(value) is not int or not low<=value<=high:
                raise ValueError('invalid '+name)
        for name,low,high in (('settle_s',.7,2.),('timeout_s',30.,600.),
                              ('min_signal_ADU',3.,30.),('noise_sigma',5.,12.),
                              ('validation_p95_px',.1,1.),('validation_max_px',.1,2.),
                              ('return_drift_px',.1,1.)):
            if not np.isfinite(getattr(self,name)) or not low<=getattr(self,name)<=high:
                raise ValueError('invalid '+name)
        if self.validation_max_px<self.validation_p95_px:
            raise ValueError('max threshold below p95')
        return self

    def groups(self):
        self.validate()
        groups=[dict(name='dark_start',split='dark',xy=None),
                dict(name='pilot_center',split='preflight',xy=[300,300])]
        if self.mode=='grid':
            # Keep even the largest permitted disk inside the task field.
            edge=max(32,16+self.radius)
            fit_positions=np.linspace(edge,600-edge,5).astype(int)
            held_out=(fit_positions[:-1]+fit_positions[1:])//2
            for split,positions in (('fit',fit_positions),('validation',held_out)):
                for row,y in enumerate(positions):
                    for col,x in enumerate(positions):
                        groups.append(dict(name=f'{split}_{row}_{col}',split=split,
                                           xy=[int(x),int(y)]))
        groups += [dict(name='return_center',split='return',xy=[300,300]),
                   dict(name='dark_end',split='dark',xy=None)]
        return groups

    def catalog(self):
        value=dict(schema=1,plan=asdict(self),groups=self.groups(),
                   DMD_coordinates='600x600 VIDEO canvas; not native micromirrors',
                   CCD_coordinates='native full sensor, x right / y down',
                   SLM='canonical full-panel gray zero; no LUT; one hardware horizontal flip',
                   fit_and_validation_disjoint=True,uses_ASM=False,
                   no_per_pattern_registration=True,hardware_synchronized=False,
                   coverage='declared point centres only; no whole-field guarantee',
                   dark_return_RMS_limit_ADU=self.min_signal_ADU,
                   release_approved=False)
        value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return value


def point_pattern(xy,radius=8):
    if type(radius) is not int or not 4<=radius<=24:
        raise ValueError('invalid point radius')
    image=np.zeros((600,600),np.uint8)
    if xy is None:return image
    p=np.asarray(xy)
    if p.shape!=(2,) or not np.isfinite(p).all() or not np.equal(p,np.floor(p)).all():
        raise ValueError('integer video coordinates required')
    x,y=map(int,p)
    if min(x,y)<12+radius or max(x,y)>587-radius:
        raise ValueError('point outside task field margin')
    yy,xx=np.ogrid[:600,:600]
    image[(xx-x)**2+(yy-y)**2<=radius**2]=255
    return image


def detect_point(light,dark,active,plan):
    """Weighted centroid of ONE isolated resolved component; reject ambiguity."""
    plan.validate()
    light=np.asarray(light,np.float32);dark=np.asarray(dark,np.float32)
    if (light.ndim!=2 or light.shape!=dark.shape or not np.isfinite(light).all()
            or not np.isfinite(dark).all() or light.shape!=(int(active['Height']),int(active['Width']))):
        raise ValueError('matching finite native CCD planes required')
    difference=light-dark
    background=float(np.median(difference))
    noise=float(1.4826*np.median(np.abs(difference-background)))
    threshold=max(plan.min_signal_ADU,plan.noise_sigma*noise)
    signal=difference-background
    count,labels,stats,_=cv2.connectedComponentsWithStats((signal>=threshold).astype(np.uint8),8)
    candidates=[]
    for index in range(1,count):
        x,y,w,h,area=map(int,stats[index])
        if area<plan.min_area:continue
        support=labels[y:y+h,x:x+w]==index
        energy=float(signal[y:y+h,x:x+w][support].sum())
        candidates.append((energy,index,(x,y,w,h,area)))
    candidates.sort(reverse=True)
    if not candidates:raise ValueError(f'单点信号不足：阈值{threshold:.2f} ADU，最小面积{plan.min_area}')
    energy,index,(x,y,w,h,area)=candidates[0]
    if len(candidates)>1 and candidates[1][0]>.2*energy:
        raise ValueError('单点出现多个显著光斑；不自动选一个伪造位置')
    height,width=light.shape;margin=plan.edge_margin
    if x<margin or y<margin or x+w>width-margin or y+h>height-margin:
        raise ValueError('单点响应贴相机边界；扩大视场或先调整光路')
    if area>.05*light.size:raise ValueError('单点响应面积异常大')
    local=signal[y:y+h,x:x+w];support=labels[y:y+h,x:x+w]==index
    yy,xx=np.indices(local.shape)
    centres=[]
    for factor in (1.,1.25,1.5):
        weights=np.where(support & (local>=threshold*factor),local,0.)
        mass=float(weights.sum())
        if not mass:raise ValueError('单点峰值过弱，无法检查阈值稳定性')
        centres.append(np.array([x+float((weights*xx).sum()/mass),y+float((weights*yy).sum()/mass)]))
    spread=float(max(np.linalg.norm(c-centres[0]) for c in centres))
    if spread>1.:raise ValueError(f'单点中心对阈值不稳定：{spread:.2f} CCD像素')
    offset=np.array([int(active['OffsetX']),int(active['OffsetY'])])
    return dict(center_sensor=(centres[0]+offset).tolist(),center_roi=centres[0].tolist(),
                bbox_roi=[x,y,w,h],area_px=area,peak_ADU=float(local.max()),
                energy_above_threshold_ADU=energy,threshold_ADU=threshold,
                noise_MAD_ADU=noise,background_change_ADU=background,
                threshold_centroid_spread_px=spread,
                center_precision_not_proven_by_decimal_digits=True)


def fit_geometry(records,plan):
    """Fit ONLY designated points. Validation NEVER selects or refits a model."""
    expected=[g for g in plan.groups() if g['xy'] is not None]
    if len(records)!=len(expected) or [r['name'] for r in records]!=[g['name'] for g in expected]:
        raise ValueError('incomplete, reordered or duplicated point records')
    for record,group in zip(records,expected):
        if record['split']!=group['split'] or record['xy']!=group['xy']:
            raise ValueError('point roles or coordinates changed after declaration')
    measured=np.asarray([r['detection']['center_sensor'] for r in records],float)
    if measured.shape!=(len(records),2) or not np.isfinite(measured).all():
        raise ValueError('invalid measured centres')
    drift=float(np.linalg.norm(measured[-1]-measured[0]))
    result=dict(mode=plan.mode,return_drift_px=drift,return_passed=bool(drift<=plan.return_drift_px),
                spatial_geometry_validated=False,frequency_geometry_validated=False,
                LUT_validated=False,release_approved=False,records=records)
    if plan.mode=='pilot':return result
    source=np.asarray([r['xy'] for r in records],float)
    design=np.c_[source,np.ones(len(source))]
    fit=np.asarray([r['split']=='fit' for r in records]);validation=np.asarray([r['split']=='validation' for r in records])
    if fit.sum()!=25 or validation.sum()!=16 or np.linalg.matrix_rank(design[fit])!=3:
        raise ValueError('invalid fit/validation partition')
    matrix=np.linalg.lstsq(design[fit],measured[fit],rcond=None)[0].T
    if abs(np.linalg.det(matrix[:,:2]))<1e-8:raise ValueError('degenerate spatial mapping')
    prediction=design@matrix.T;vectors=measured-prediction;errors=np.linalg.norm(vectors,axis=1)
    p95=float(np.quantile(errors[validation],.95));maximum=float(errors[validation].max())
    result.update(matrix_DMD_video_to_CCD_sensor=matrix.tolist(),
                  fit_rms_px=float(np.sqrt(np.mean(errors[fit]**2))),
                  validation_p95_px=p95,validation_max_px=maximum,
                  predicted_centres=prediction.tolist(),residual_vectors_px=vectors.tolist(),
                  residual_norms_px=errors.tolist(),fit_count=25,validation_count=16,
                  spatial_geometry_validated=bool(drift<=plan.return_drift_px
                      and p95<=plan.validation_p95_px and maximum<=plan.validation_max_px))
    return result


def geometry_overlay(result,active):
    """Native-coordinate diagram, not a warped CCD image. Arrows magnified x20."""
    width,height=int(active['Width']),int(active['Height'])
    scale=min(1000/width,750/height)
    image=np.full((max(1,round(height*scale)),max(1,round(width*scale)),3),20,np.uint8)
    offset=np.array([int(active['OffsetX']),int(active['OffsetY'])])
    predicted=result.get('predicted_centres')
    for i,record in enumerate(result['records']):
        real=np.array(record['detection']['center_sensor'])-offset
        pos=tuple(np.round(real*scale).astype(int))
        color=(40,220,40) if record['split']=='fit' else (255,170,40)
        cv2.circle(image,pos,4,color,1)
        if predicted is not None:
            target=np.asarray(predicted[i])-offset
            end=target+20*(real-target)
            cv2.arrowedLine(image,tuple(np.round(target*scale).astype(int)),
                            tuple(np.round(end*scale).astype(int)),(240,70,70),1,tipLength=.3)
    cv2.putText(image,'DMD -> CCD; green=fit, orange=held-out; arrows x20',
                (10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
    return image
