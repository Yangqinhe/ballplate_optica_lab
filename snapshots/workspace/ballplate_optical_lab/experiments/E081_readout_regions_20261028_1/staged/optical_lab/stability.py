"""Fixed-input repeatability: pure planning/math; explicit file-only export.

No Qt, device SDK, process discovery or hardware action on import. Independent
visits, not constituent RAW frames, are the samples for the across-visit table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from .effective_frame import FilterSpec, EffectiveWindow


@dataclass(frozen=True)
class Settings:
    frames: int = 30
    repeats: int = 6
    switch_inputs: bool = False
    settle_seconds: float = 1.5
    interval_seconds: float = 2.0
    software_gain: float = 1.0
    black_ADU: float = 12.5
    gamma: float = .5
    maximum_seconds: float = 300.0
    input_ids: tuple = ()
    filter_mode: str = 'mean'
    filter_alpha: float = .5

    def validate(self):
        if (not isinstance(self.input_ids, tuple) or len(self.input_ids)>16
                or len(set(self.input_ids))!=len(self.input_ids)
                or any(not isinstance(v,str) or not v.replace('_','').isalnum() for v in self.input_ids)):
            raise ValueError('测试输入必须为1–16个唯一名称')
        FilterSpec(self.filter_mode,self.frames,self.filter_alpha).validate()
        if type(self.repeats) is not int or not 3 <= self.repeats <= 20:
            raise ValueError('每种输入至少3次、最多20次独立回访')
        if type(self.switch_inputs) is not bool:
            raise ValueError('换图必须为明确布尔选项')
        for name, low, high in (('settle_seconds', 1.5, 15), ('interval_seconds', 0, 15),
                               ('software_gain', 1, 128), ('black_ADU', 0, 255),
                               ('gamma', .1, 3), ('maximum_seconds', 5, 300)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('参数超出范围：'+name)
        return self


def schedule(settings):
    settings.validate()
    labels = settings.input_ids or (('A','B') if settings.switch_inputs else ('A',))
    return [dict(input_id=label, visit=index+1)
            for index in range(settings.repeats) for label in labels]


def mapping(raw, settings):
    """Map after shared RAW filtering; preview and processed comparison match."""
    settings.validate()
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any() or (a > 255).any():
        raise ValueError('finite native Mono8-valued plane required')
    unit = np.clip((a-settings.black_ADU)*settings.software_gain/255., 0, 1)
    return np.rint(255.*unit**settings.gamma).astype(np.uint8)


REGION_COUNTS = (8,16,32,64)


def feature_vector(row, domain='raw'):
    """Generic vectors; archived 8-region records remain readable."""
    key={'raw':'raw','processed':'processed','within':'within_window_RAW',
         'within_sd':'within_window_RAW_SD'}[domain]
    return row[key] if key in row else row[key+'8']


def region_count(visits):
    counts={len(feature_vector(row)) for row in visits}
    if len(counts)!=1 or next(iter(counts)) not in REGION_COUNTS:
        raise ValueError('读出分区数不一致；不能混算不同布局')
    count=next(iter(counts))
    for row in visits:
        if row.get('region_count',count)!=count or len(feature_vector(row,'processed'))!=count:
            raise ValueError('分区元数据与向量维度不一致')
    return count


def validate_weights(weights, shape):
    w = np.asarray(weights)
    if (w.ndim!=3 or w.shape[0] not in REGION_COUNTS or w.shape[1:]!=tuple(shape)
            or not np.isin(w,(0,1)).all() or not np.all(w.sum(0)==1)):
        raise ValueError('8/16/32/64区必须完整覆盖当前ROI、无重叠且为二值mask')
    if np.any(w.sum((1,2))==0):raise ValueError('读出分区不能包含空区域')
    return w


def readout(raws, weights, settings):
    """One pass over RAWs; memory is O(ROI pixels), not O(N × float image)."""
    settings.validate()
    w = np.asarray(weights)
    validate_weights(w, w.shape[1:])
    shape = w.shape[1:]
    count = w.shape[0]
    indices = [np.flatnonzero(mask) for mask in w]
    labels = w.argmax(0).ravel()
    counts = np.bincount(labels, minlength=count)
    total = np.zeros(shape, np.uint32)
    spec=FilterSpec(settings.filter_mode,settings.frames,settings.filter_alpha)
    accumulator=EffectiveWindow(spec);effective=None
    any_clipped = np.zeros(shape, bool)
    peaks = np.zeros(count, np.uint8)
    clips = np.zeros(count, np.int64); near = np.zeros(count, np.int64)
    max_frame_clips = np.zeros(count, np.int64)
    single = []
    for frame in raws:
        a = np.asarray(frame)
        if a.shape != shape or a.dtype != np.uint8 or len(single) >= settings.frames:
            raise ValueError('one complete independent N-frame native uint8 batch required')
        total += a
        effective=accumulator.push(a)
        clipped = a == 255
        any_clipped |= clipped
        clip8 = np.bincount(labels[clipped.ravel()], minlength=count)
        clips += clip8
        near += np.bincount(labels[(a >= 240).ravel()], minlength=count)
        max_frame_clips = np.maximum(max_frame_clips, clip8)
        for k in range(count):
            peaks[k] = max(peaks[k], a.ravel()[indices[k]].max())
        single.append(np.bincount(labels, weights=a.ravel(), minlength=count))
    if len(single) != settings.frames:
        raise ValueError('one complete independent N-frame native uint8 batch required')
    mean = total.astype(np.float32) / np.float32(settings.frames)
    if effective is None:raise ValueError('独立有效帧未完成')
    processed = mapping(effective, settings)
    raw8 = np.bincount(labels, weights=effective.ravel(), minlength=count)
    processed8 = np.bincount(labels, weights=processed.ravel(), minlength=count)
    regions = []
    for k in range(count):
        yy, xx = np.where(any_clipped & w[k].astype(bool))
        observations = int(counts[k]) * settings.frames
        regions.append(dict(region=k, max_ADU=int(peaks[k]),
            clipped_pixel_observations=int(clips[k]), pixel_observations=observations,
            clipped_percent=100*float(clips[k])/observations,
            max_frame_clipped_percent=100*float(max_frame_clips[k])/int(counts[k]),
            unique_clipped_pixels=len(xx),
            near_saturation_pixel_observations=int(near[k]),
            near_saturation_percent=100*float(near[k])/observations,
            clipped_bbox_xyxy=([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None)))
    result = dict(region_count=count,raw=raw8.tolist(), processed=processed8.tolist(), brightness_regions=regions,
                effective_filter=asdict(spec),primary_domain='unmapped_RAW_effective_frame',
                arithmetic_mean=np.bincount(labels,weights=mean.ravel(),minlength=count).tolist(),
                brightness_warning=bool(peaks.max()>=240),
                within_window_RAW_SD=np.asarray(single).std(0, ddof=1 if settings.frames>1 else 0).tolist(),
                within_window_RAW=np.asarray(single).tolist(),
                max_ADU=int(peaks.max()), clipped_pixels=int(clips.sum()),
                frames=len(single))
    if count==8:
        for key in ('raw','processed','arithmetic_mean','within_window_RAW_SD','within_window_RAW'):
            result[key+'8']=result[key]  # Explicit legacy aliases only for actual eight-region data.
    return result, effective, processed


def column_summary(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('至少三次独立回访，且读数必须有限')
    mean = float(x.mean()); sd = float(x.std(ddof=1))
    denominator = abs(mean)
    qualified = denominator > 1.
    return dict(mean=mean, sample_SD=sd, minimum=float(x.min()), maximum=float(x.max()),
                CV_percent=100*sd/denominator if qualified else None,
                peak_to_peak_percent=100*float(np.ptp(x))/denominator if qualified else None,
                first_to_last_percent=100*float(x[-1]-x[0])/denominator if qualified else None,
                relative_status='DEFINED' if qualified else 'UNKNOWN_SMALL_MEAN')


def summary(features, *, allow_signed=False):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] not in REGION_COUNTS or len(x) < 3 or not np.isfinite(x).all() or (not allow_signed and (x < 0).any()):
        raise ValueError('每种输入至少三次独立、非负且维度一致的区域读数')
    count=x.shape[1]
    rows = [dict(region=k, **column_summary(x[:, k])) for k in range(count)]
    mean = x.mean(0); norm = float(np.linalg.norm(mean))
    rms = float(np.sqrt(np.mean(np.sum((x-mean)**2, axis=1))))
    pair = max(float(np.linalg.norm(a-b)) for i, a in enumerate(x) for b in x[i+1:])
    vector_rms = 100*rms/norm if norm > 1 else None
    maximum_pair = 100*pair/norm if norm > 1 else None
    cvs = [r['CV_percent'] for r in rows]
    gate = ('UNKNOWN' if vector_rms is None or any(c is None for c in cvs) else
            'RESEARCH_PASS' if vector_rms <= 1 and maximum_pair <= 3 and max(cvs) <= 3 else 'RESEARCH_FAIL')
    return dict(region_count=count,independent_visits=len(x), regions=rows, relative_vector_RMS_percent=vector_rms,
                maximum_pair_L2_percent=maximum_pair, maximum_region_CV_percent=(max(cvs) if all(c is not None for c in cvs) else None),
                reference_gate=gate, gate_definition='RMS≤1%; max pair≤3%; each CV≤3%; descriptive research only',
                control_qualified=False, physical_cause_identified=False)


def analyze(visits, settings):
    expected = schedule(settings)
    if len(visits) != len(expected) or any(any(v[k] != e[k] for k in ('input_id', 'visit')) for v, e in zip(visits, expected)):
        raise ValueError('完整预声明顺序未完成；失败前缀不能生成正式波动判定')
    count=region_count(visits)
    output = {}; photometry = {}
    for label in dict.fromkeys(v['input_id'] for v in expected):
        selected = [v for v in visits if v['input_id'] == label]
        output[label] = {domain: summary([feature_vector(v,domain) for v in selected]) for domain in ('raw', 'processed')}
        complete = all('brightness_regions' in v for v in selected)
        clipped = any(v.get('clipped_pixels', 0)>0 for v in selected)
        near = any(v.get('max_ADU', 0)>=240 for v in selected)
        regions = []
        if complete:
            for k in range(count):
                rows = [v['brightness_regions'][k] for v in selected]
                total = sum(r['pixel_observations'] for r in rows)
                clips = sum(r['clipped_pixel_observations'] for r in rows)
                regions.append(dict(region=k,max_ADU=max(r['max_ADU'] for r in rows),
                    clipped_pixel_observations=clips,pixel_observations=total,
                    clipped_percent=100*clips/total,
                    max_frame_clipped_percent=max(r['max_frame_clipped_percent'] for r in rows),
                    near_saturation_percent=100*sum(r['near_saturation_pixel_observations'] for r in rows)/total))
        photometry[label] = dict(status='CLIPPED' if clipped else 'NEAR_SATURATION' if near else 'NO_THRESHOLD_HIT' if complete else 'UNKNOWN_LEGACY',
                                regions=regions,per_region_available=complete,acquisition_continues=True)
        if clipped:
            for stats in output[label].values():
                stats['numerical_repeatability_gate'] = stats['reference_gate']
                stats['reference_gate'] = 'SATURATED_UNQUALIFIED'
    return dict(schema='optical-lab-readout-stability-results-3', region_count=count, settings=asdict(settings), inputs=output,
                effective_filter=asdict(FilterSpec(settings.filter_mode,settings.frames,settings.filter_alpha)),
                visits=visits, photometry=photometry, different_inputs_never_pooled=True, each_visit_history_reset=True,
                control_qualified=False, units='intensity-code sums; not calibrated optical power')


def file_ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def analyze_archive(reference, weights, settings, item, expected_active):
    """Read a sealed original sampler archive on a file-only worker thread."""
    def pinned(ref):
        p = Path(ref['path'])
        if not p.is_absolute() or p.is_symlink() or file_ref(p) != ref:
            raise ValueError('封存文件SHA不一致：'+str(p))
        return p
    path = pinned(reference)
    batch = json.loads(path.read_text())
    if (batch.get('completed') is not True or batch.get('error') is not None
            or batch.get('source_recheck_error') is not None
            or batch.get('expected_active') != expected_active
            or any(batch.get(k) != settings.frames for k in ('requested_frames', 'captured_frames', 'valid_frames'))
            or len(batch.get('frames', [])) != settings.frames):
        raise ValueError('失败／不完整／参数变化的采样窗，不能用作有效帧')
    stamps = []; ids = []; host_stamps = []
    def raw_frames():
      for index, row in enumerate(batch['frames']):
        if row['index'] != index or row['valid'] is not True:
            raise ValueError('无效RAW顺序')
        rp = pinned(row['raw']); mp = pinned(row['metadata'])
        if rp.parent != path.parent or mp.parent != path.parent:
            raise ValueError('RAW跨采样窗')
        meta = json.loads(mp.read_text()); raw = np.load(rp, allow_pickle=False)
        if (meta.get('raw') != row['raw'] or meta.get('valid') is not True
                or meta.get('active_readback_before') != expected_active
                or meta.get('active_readback_after') != expected_active
                or hashlib.sha256(raw.tobytes()).hexdigest() != meta.get('raw_pixel_sha256')):
            raise ValueError('原始帧与元数据／相机参数不一致')
        stamps.append(int(meta['original_metadata']['TimeStamp']))
        ids.append(int(meta['original_metadata']['FrameID']))
        host_stamps.append(meta['original_metadata'].get('host_received_unix_ns'))
        yield raw
    row, mean, processed = readout(raw_frames(), weights, settings)
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('CCD时间戳未严格递增')
    # TimeStamp is in ns in this camera adapter, not a host preview timestamp.
    duration = (stamps[-1]-stamps[0])/1e9
    row.update(item, archive=reference, sample_span_seconds=duration,
               observed_rate_hz=(settings.frames-1)/duration if duration > 0 else None,
               first_frame_id=ids[0], last_frame_id=ids[-1],
               mean_host_received_ns=(int(sum(host_stamps)//len(host_stamps)) if all(v is not None for v in host_stamps) else None),
               contiguous_native_frames_claimed=False)
    return row, mean, processed


def export_brightness_maps(reference, directory):
    """Exact ROI-relative hit locations; RAW unchanged. File-worker only."""
    p = Path(reference['path'])
    if file_ref(p) != reference: raise ValueError('archive SHA mismatch')
    batch = json.loads(p.read_text()); counts = None
    for row in batch['frames']:
        rp = Path(row['raw']['path'])
        if file_ref(rp) != row['raw']: raise ValueError('RAW SHA mismatch')
        a = np.load(rp, allow_pickle=False)
        if counts is None: counts = np.zeros((2,*a.shape),dtype=np.uint16)
        counts[0] += a>=240; counts[1] += a==255
    refs={}
    for name, plane in zip(('near_saturation_hits','clipped_hits'),counts):
        dest=Path(directory)/(name+'.npy')
        with dest.open('xb') as f:np.save(f,plane,allow_pickle=False)
        refs[name]=file_ref(dest)
    return dict(files=refs,coordinates='native ROI pixels; add camera OffsetX/OffsetY for sensor coordinates',
                units='number of RAW frames hitting threshold at each pixel')


def region_figure(result, label, domain):
    """Eight simultaneous traces for ONE input/domain; no device or Qt access.

    Relative deviations use each region's own across-visit mean, not a shared
    optical-power scale. Small-mean regions remain visibly UNKNOWN.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if label not in result['inputs'] or domain not in result['inputs'][label]:
        raise ValueError('explicit result input/domain required')
    selected = [v for v in result['visits'] if v['input_id'] == label]
    x = np.asarray([feature_vector(v,domain) for v in selected], dtype=np.float64)
    stats = summary(x, allow_signed=True)
    means = x.mean(0); valid = means > 1
    deviations = np.full_like(x, np.nan)
    deviations[:, valid] = 100*(x[:, valid]/means[valid]-1)
    extent = max(1., float(np.nanmax(np.abs(deviations)))*1.15) if valid.any() else 1.
    count=x.shape[1];cols=4 if count<=16 else 8;rows=count//cols
    fig = Figure(figsize=(12 if cols==4 else 18,2.5*rows+1), constrained_layout=True); FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        cv = stats['regions'][k]['CV_percent']
        ax.set_title(f'R{k} | CV {cv:.2f}%' if cv is not None else f'R{k} | UNKNOWN')
        if valid[k]:
            ax.plot([v['visit'] for v in selected], deviations[:, k], 'o-', color='#2878B5', markersize=4)
        else:
            ax.text(.5, .5, 'Mean too small', ha='center', transform=ax.transAxes)
        ax.axhline(0, color='#777777', lw=.6); ax.grid(alpha=.2)
        ax.set_ylim(-extent, extent)
        ax.xaxis.get_major_locator().set_params(integer=True)
    frames = selected[0].get('frames',result['settings']['frames'])
    geometry=result.get('input_geometry',{}).get(label,{})
    size=f' | ball dilation R={geometry["radius_raw_px"]} RAW96 px' if geometry.get('kind')=='four_ball_history' else ''
    fig.suptitle(f'Input {label} | {domain} | {frames} target RAW / visit | all {count} regions'+size)
    fig.supxlabel('Independent visit (history reset per visit)')
    fig.supylabel('Deviation from each region mean (%)')
    return fig


def export(result, directory):
    """File-only background work. New result directory, no overwriting RAW."""
    out = Path(directory); out.mkdir(exist_ok=False)
    with (out/'RESULT.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    keys = ['input', 'domain', 'region', 'mean', 'sample_SD', 'CV_percent', 'peak_to_peak_percent', 'first_to_last_percent']
    with (out/'regions.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for label, domains in result['inputs'].items():
            for domain, info in domains.items():
                for row in info['regions']:
                    writer.writerow(dict(input=label, domain=domain, **{k: row[k] for k in keys[2:]}))
    with (out/'brightness_regions.csv').open('x', encoding='utf-8-sig', newline='') as f:
        keys=['input','region','max_ADU','clipped_pixel_observations','pixel_observations','clipped_percent','max_frame_clipped_percent','near_saturation_percent']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for label,p in result.get('photometry',{}).items():
            for row in p['regions']:writer.writerow(dict(input=label,**row))
    # All eight regions receive the same chart and table treatment.
    for label in result['inputs']:
        for domain in result['inputs'][label]:
            chart = region_figure(result, label, domain)
            for ext in ('png', 'pdf'):
                chart.savefig(out/(f'stability_{label}_{domain}.'+ext), dpi=160)
                if label == 'A' and domain == 'processed':
                    chart.savefig(out/('stability.'+ext), dpi=160)
    text = ['# 读出稳定性测试', '', '每次有效帧按主界面滤波参数独立处理N张RAW；显示映射另做处理后对照。不同输入分别统计，不混合。',
            '指标不代表光子SNR、物理根因、全天稳定或闭环资格。',
            '主图显示本轮所有区域相对各自均值的波动，各区共用纵轴尺度；A/B及RAW/处理后分别出图。',
            '亮度告警不终止采样或统计；含255ADU数据标记SATURATED_UNQUALIFIED，即使数值CV为零也不能用于证明部署达标。各区亮度见brightness_regions.csv，逐窗像素位置见MEAN目录的clipped_hits.npy。', '']
    for label, domains in result['inputs'].items():
        d = domains['processed']; text += [f'## 输入{label}', '',
            f'独立回访{d["independent_visits"]}次；相对向量RMS={d["relative_vector_RMS_percent"]}%；研究参考判定={d["reference_gate"]}。', '',
            '| Region | 均值 | 样本SD | CV% | 峰峰差/均值% |', '|---|---:|---:|---:|---:|']
        for r in d['regions']:
            text.append(f'| {r["region"]} | {r["mean"]:.4f} | {r["sample_SD"]:.4f} | {r["CV_percent"]} | {r["peak_to_peak_percent"]} |')
        text += ['']
    (out/'总结.md').write_text('\n'.join(text), encoding='utf-8')
    export_assessment(result, out)
    hashes = {p.name: file_ref(p)['sha256'] for p in sorted(out.iterdir()) if p.is_file()}
    with (out/'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(hashes, f, indent=2)
    return out


CONTROL_LIMITS = {
    'MAE_rad': .06, 'p95_error_L2_rad': .18, 'center_max_L2_rad': .05,
    'center_max_pair_rad': .04, 'same_input_pair_p95_rad': .04,
    'same_input_pair_max_rad': .08,
}


def head_weights(head):
    return np.asarray(head.get('weights',head.get('weights_8x2')),float)


def load_head(path):
    """Explicit file-only K→2 linear head; legacy eight heads stay readable."""
    ref=file_ref(path);head=json.loads(Path(ref['path']).read_text())
    w=head_weights(head);b=np.asarray(head.get('bias_2'),float)
    if (w.ndim!=2 or w.shape[0] not in REGION_COUNTS or w.shape[1]!=2
            or head.get('region_count',w.shape[0])!=w.shape[0]
            or head.get('parameter_count')!=2*w.shape[0]+2 or b.shape!=(2,)
            or not np.isfinite(w).all() or not np.isfinite(b).all()
            or 'RAW' not in head.get('feature_domain','') or 'fixed_camera' not in head
            or not head.get('optical_mask_sha256') or not head.get('readout_mask')):
        raise ValueError('需要8/16/32/64→2 RAW线性头及相机/SLM/分区元数据')
    return dict(reference=ref,head=head)


def assess_controls(visits, domains, head_record, labels, context):
    """Snapshot head only; labels score outputs, never modify those outputs."""
    if head_record is None:
        return dict(status='UNKNOWN_NO_HEAD', domains={}, deployment_qualified=False)
    head = head_record['head']; w = head_weights(head); b = np.asarray(head['bias_2'])
    count=region_count(visits)
    if w.shape!=(count,2):
        return dict(status='UNKNOWN_HEAD_DIMENSION',domains={},deployment_qualified=False,
            expected_regions=count,loaded_regions=w.shape[0],
            reason=f'当前{count}区，载入头为{w.shape[0]}→2；需适配对应线性头，读出稳定性仍可评估')
    mismatches = []
    filters={json.dumps(v.get('effective_filter',dict(mode='mean',frames=v['frames'],alpha=.5)),sort_keys=True) for v in visits}
    for encoded in filters:
        spec=json.loads(encoded);expected=head.get('effective_filter',dict(mode='mean',frames=head.get('mean_frames',50),alpha=.5))
        if spec['mode']!=expected['mode'] or spec['frames']!=expected['frames'] or (spec['mode']=='ema' and spec['alpha']!=expected.get('alpha')):
            mismatches.append('有效帧滤波与候选头训练配置不匹配，不能沿用均值头资格')
    if not np.isclose(context.get('dmd_scale',1.),head.get('dmd_scale',1.),rtol=0,atol=1e-8):
        mismatches.append('DMD缩放不同，需核验候选头的尺度配置')
    same_mask=head['readout_mask']['sha256']==context['mask_sha256']
    if not same_mask and context.get('readout_mask_ref'):
        # Equal binary weights with different NPY dtypes/headers are equivalent.
        try:
            left,right=head['readout_mask'],context['readout_mask_ref']
            same_mask=(file_ref(left['path'])==left and file_ref(right['path'])==right
                       and np.array_equal(np.load(left['path'],allow_pickle=False),np.load(right['path'],allow_pickle=False)))
        except (OSError,ValueError):pass
    if not same_mask:mismatches.append('读出mask不同或未核验')
    if head['optical_mask_sha256'] != context['slm_sha256']: mismatches.append('SLM记录不匹配或面板封装未核验')
    for k, v in head['fixed_camera'].items():
        if context['camera_actual'].get(k) != v: mismatches.append('相机 '+k)
    expected_N=head.get('effective_filter',{}).get('frames',head.get('mean_frames',50))
    if any(v['frames'] != expected_N for v in visits): mismatches.append(f'有效帧RAW数不同于候选头N{expected_N}')
    for name,geometry in (context.get('input_geometry') or {}).items():
        if geometry.get('fill_enclosed_holes',False)!=head.get('fill_enclosed_holes',False):
            mismatches.append(f'输入{name}填孔编码与候选头不同；需重新验证/适配，当前只诊断')
        if name=='R':continue
        if geometry.get('kind')=='four_ball_history' and geometry.get('radius_raw_px')!=head.get('ball_radius_raw_px',2):
            mismatches.append(f'输入{name}球尺寸R={geometry.get("radius_raw_px")}，不同于候选头编码；需重新适配，当前只诊断')
        elif geometry.get('kind')=='unknown':mismatches.append(f'输入{name}球尺寸未知，不能继承旧头资格')
    result = {}
    for domain in domains:
        values = np.asarray([feature_vector(v,domain) for v in visits]); actions = values@w+b
        targets = []; errors = []; centers = []; pairs = []; center_pairs = []
        groups = list(dict.fromkeys(v['input_id'] for v in visits))
        for i, row in enumerate(visits):
            label = labels.get(row['input_id'], {}); target = label.get('teacher_action_rad')
            # Synthetic references sometimes carry zero-valued loader placeholders.
            if label.get('state_fields_are_loader_placeholders') or label.get('input_kind') == 'calibration_probe_not_MuJoCo_state':
                target = None
            if target is not None:
                a = np.asarray(target, float)
                if a.shape != (2,) or not np.isfinite(a).all(): raise ValueError('目标控制标签无效')
                errors.append(actions[i]-a)
            targets.append(target)
            is_center = (target is not None and label.get('position_xy_m') == [0.,0.]
                         and label.get('velocity_xy_m_s') == [0.,0.])
            if is_center: centers.append(i)
        per_input = {}
        for label in groups:
            indices = [i for i,v in enumerate(visits) if v['input_id'] == label]
            pairs.extend(float(np.linalg.norm(actions[i]-actions[j])) for n,i in enumerate(indices) for j in indices[n+1:])
            group_actions=actions[indices]
            per_input[label]=dict(repeat_RMS_L2_rad=float(np.sqrt(np.mean(np.sum((group_actions-group_actions.mean(0))**2,axis=1)))),
                mean_action_rad=group_actions.mean(0).tolist(),visits=len(indices))
        center_pairs = [float(np.linalg.norm(actions[i]-actions[j])) for n,i in enumerate(centers) for j in centers[n+1:]]
        e = np.asarray(errors)
        metrics = dict(MAE_rad=float(np.abs(e).mean()) if len(e) else None,
            p95_error_L2_rad=float(np.percentile(np.linalg.norm(e,axis=1),95)) if len(e) else None,
            center_max_L2_rad=float(np.linalg.norm(actions[centers],axis=1).max()) if centers else None,
            center_max_pair_rad=max(center_pairs) if center_pairs else None,
            same_input_pair_p95_rad=float(np.percentile(pairs,95)) if pairs else None,
            same_input_pair_max_rad=max(pairs) if pairs else None)
        # Sensitivity separates absolute optical fluctuations from head amplification.
        contributions = []
        for label in groups:
            x = values[[i for i,v in enumerate(visits) if v['input_id']==label]]
            contributions.extend((x-x.mean(0))[:,:,None]*w[None,:,:])
        reasons = list(mismatches)
        if domain != 'raw': reasons.append('新读出域未用于此头训练；仅敏感性诊断')
        result[domain] = dict(actions_rad=actions.tolist(), teacher_actions_rad=targets,
            input_ids=[v['input_id'] for v in visits], metrics=metrics,
            gates={k:dict(value=v, limit=CONTROL_LIMITS[k], status='UNKNOWN' if v is None else
                   'LOCAL_NUMERIC_PASS' if v<=CONTROL_LIMITS[k] else 'FAIL') for k,v in metrics.items()},
            region_action_RMS_rad=np.sqrt(np.mean(np.asarray(contributions)**2,axis=0)).tolist(),
            compatibility='DIAGNOSTIC_ONLY' if reasons else 'MATCHED_RAW_CONTEXT', context_warnings=reasons,
            evaluated_target_rows=len(errors), unique_inputs=len(groups),per_input=per_input)
    return dict(status='STATIC_DIAGNOSTIC', head=head_record['reference'], domains=result,
        limits=CONTROL_LIMITS, engineering_limits_not_stability_theorem=True,
        coverage='A/B手动子集；非完整位置/速度、时间块及闭环验收', deployment_qualified=False,
        no_action_clipping=True, no_head_fit=True)


def assess(visits, settings, head_record=None, labels=None, context=None):
    """Descriptive research gates, never an automatic control authorization.

    Contrast between INPUTS is compared with ACROSS-VISIT scatter. A large
    background or N constituent frames must not masquerade as independent
    information. Ground-truth labels are used for scoring/coverage only.
    """
    labels, context = labels or {}, context or {}
    result = analyze(visits, settings)
    result['input_geometry'] = context.get('input_geometry', {})
    result['control_analysis'] = assess_controls(visits, ('raw',), head_record, labels, context)
    count=result['region_count']
    groups = list(result['inputs'])
    features = {name: np.asarray([feature_vector(v) for v in visits if v['input_id']==name], float) for name in groups}
    diagnostics = {}
    for name, x in features.items():
        rows = [v for v in visits if v['input_id']==name]
        mean = x.mean(0); valid = np.abs(mean)>1
        sd = x.std(0, ddof=1)
        within = np.sqrt(np.mean(np.asarray([feature_vector(v,'within_sd') for v in rows])**2, axis=0))
        corr = np.full((count,count), np.nan)
        for i in range(count):
            for j in range(count):
                if sd[i]>1e-12 and sd[j]>1e-12:
                    corr[i,j] = np.corrcoef(x[:,i], x[:,j])[0,1]
        times = [v.get('mean_host_received_ns') for v in rows]
        t = ((np.asarray(times,dtype=np.float64)-times[0])/1e9
             if all(isinstance(v,int) for v in times) else None)
        span = float(t[-1]) if t is not None else None
        trend = np.zeros(count) if t is None or span<=0 else np.polyfit(t-t.mean(), x-x.mean(0), 1)[0]*span
        regions = []
        for k in range(count):
            delta = []
            for row in rows:
                trace = np.asarray(feature_vector(row,'within'),float)
                if trace.shape==(row['frames'],count):
                    if len(trace)<2:continue
                    half = len(trace)//2
                    delta.append(float(trace[half:,k].mean()-trace[:half,k].mean()))
            regions.append(dict(region=k,
                within_RAW_CV_percent=float(100*within[k]/abs(mean[k])) if valid[k] and settings.frames>1 else None,
                across_mean_CV_percent=float(100*sd[k]/abs(mean[k])) if valid[k] else None,
                empirical_SD_reduction=float(within[k]/sd[k]) if sd[k]>1e-12 and settings.frames>1 else None,
                fitted_change_percent=float(100*trend[k]/abs(mean[k])) if valid[k] and t is not None else None,
                max_within_half_change_percent=float(100*max(map(abs,delta))/abs(mean[k])) if delta and valid[k] else None))
        diagnostics[name] = dict(regions=regions, observed_span_s=span,
            correlation=[[float(v) if np.isfinite(v) else None for v in row] for row in corr],
            time_unit='host seconds between independent effective-window midpoint timestamps',
            note='SD reduction is empirical, not photon SNR or a sqrt(N) independence claim')
    pairs = []
    for i,a in enumerate(groups):
        for b in groups[i+1:]:
            x,y = features[a],features[b]
            delta = x.mean(0)-y.mean(0)
            noise = float(np.sqrt(np.var(x,axis=0,ddof=1).sum()+np.var(y,axis=0,ddof=1).sum()))
            distance = float(np.linalg.norm(delta))
            ratio = distance/noise if noise>1e-12 else None
            pairs.append(dict(a=a,b=b,distance_ADU_sum=distance,repeat_scatter_ADU_sum=noise,
                separation_ratio=ratio,status='UNKNOWN_ZERO_SCATTER' if ratio is None else 'PASS' if ratio>=3 else 'FAIL',
                engineering_limit=3.,not_photon_SNR=True))
    covered = set(); ranges = [0.,0.,0.,0.]
    for name in groups:
        row = labels.get(name,{})
        if row.get('state_fields_are_loader_placeholders') or row.get('input_kind')=='calibration_probe_not_MuJoCo_state':continue
        p,v = np.asarray(row.get('position_xy_m',[])),np.asarray(row.get('velocity_xy_m_s',[]))
        if p.shape!=(2,) or v.shape!=(2,) or not np.isfinite(np.r_[p,v]).all():continue
        state = np.r_[p,v]; ranges = np.maximum(ranges,np.abs(state)).tolist()
        if np.allclose(state,0,atol=1e-10):covered.add('CENTER')
        for k,axis in enumerate(('X','Y','VX','VY')):
            if abs(state[k])>1e-8 and np.allclose(np.delete(state,k),0,atol=1e-10):
                covered.add(axis+('+' if state[k]>0 else '-'))
    required = ['CENTER','X+','X-','Y+','Y-','VX+','VX-','VY+','VY-']
    missing = [v for v in required if v not in covered]
    rows = []
    def gate(name,status,value,criterion):rows.append(dict(item=name,status=status,value=value,criterion=criterion))
    repeat_gates=[result['inputs'][g]['raw']['reference_gate'] for g in groups]
    repeat_status=('FAIL' if any(v in ('RESEARCH_FAIL','SATURATED_UNQUALIFIED') for v in repeat_gates)
                   else 'PASS' if all(v=='RESEARCH_PASS' for v in repeat_gates) else 'UNKNOWN')
    gate('RAW重复性', repeat_status,
         '；'.join(f'{g}: {result["inputs"][g]["raw"]["reference_gate"]}' for g in groups), '各输入RMS≤1%，最大两次差≤3%，各区CV≤3%')
    brightness = [result['photometry'][g]['status'] for g in groups]
    gate('传感器截顶', 'FAIL' if 'CLIPPED' in brightness else 'UNKNOWN' if 'UNKNOWN_LEGACY' in brightness else 'PASS',
         '；'.join(f'{g}: {result["photometry"][g]["status"]}' for g in groups), '0个255像素；≥240仅提示，测试不中断')
    sep_ok = bool(pairs) and all(v['status']=='PASS' for v in pairs)
    gate('输入可区分度', 'PASS' if sep_ok else 'FAIL' if any(v['status']=='FAIL' for v in pairs) else 'UNKNOWN',
         min((v['separation_ratio'] for v in pairs if v['separation_ratio'] is not None),default=None), '最差输入对：均值距离/跨次散布≥3（工程筛查）')
    gate('位置/速度覆盖', 'UNKNOWN' if missing else 'PASS', '缺少 '+', '.join(missing) if missing else '9类基础输入均覆盖',
         '中心，纯±X/±Y位置，当前位置相同的纯±VX/±VY；不是全域/留出验收')
    control = result['control_analysis']['domains'].get('raw')
    gate('候选头配置一致性', 'UNKNOWN' if control is None else 'PASS' if not control['context_warnings'] else 'FAIL',
         result['control_analysis'].get('reason','未载入候选RAW头') if control is None else '；'.join(control['context_warnings']) or '记录一致', '读出区域、SLM、ROI/相机、DMD尺度、球大小和N匹配')
    if control:
        for key,metric in control['gates'].items():
            gate(key, 'PASS' if metric['status']=='LOCAL_NUMERIC_PASS' else metric['status'],metric['value'],f'≤{metric["limit"]} rad')
    else:gate('控制映射误差','UNKNOWN',None,f'需要匹配的{2*count+2}参数（{count}→2）RAW头及独立输入控制标签')
    static_ok = all(r['status']=='PASS' for r in rows)
    gate('跨会话/独立留出复测','UNKNOWN',None,'本轮为单次有界静态实验，不能代替跨时间块、独立位置/轨迹复测')
    gate('MuJoCo闭环及扰动恢复','UNKNOWN',None,'未运行闭环，未验证边缘复位和抗扰动')
    result.update(diagnostics=diagnostics,separation_pairs=pairs,
        coverage=dict(required=required,covered=sorted(covered),missing=missing,max_abs_xy_vxy=ranges,
                      boundary_12cm_covered=ranges[0]>=.12 and ranges[1]>=.12,independent_holdout_verified=False),
        readiness=dict(gates=rows,static_screen_passed=static_ok,direct_deployment_qualified=False,
            recommendation=('静态筛查通过，可另行安排有界MuJoCo短闭环验证；不代表部署验收。' if static_ok else
                            '暂不建议直接部署：先处理FAIL项，再补UNKNOWN项。区域低波动不等于控制映射正确。'),
            numerical_limits_are_engineering_screen=True),
        accepted_raw_total=sum(v['frames'] for v in visits),input_labels=labels,
        shared_main=context.get('shared_main',{}))
    return result


def export_assessment(result, out):
    """Same eight-region style for noise, drift and separation. File worker only."""
    if 'readiness' not in result:return
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for name,info in result['diagnostics'].items():
        count=len(info['regions'])
        fig=Figure(figsize=(14 if count>16 else 12,6),constrained_layout=True);FigureCanvasAgg(fig)
        left,right=fig.subplots(1,2)
        x=np.arange(count);ticks=np.unique(np.r_[x[::max(1,count//16)],count-1])
        def values(key):return [np.nan if r[key] is None else r[key] for r in info['regions']]
        left.bar(x-.18,values('within_RAW_CV_percent'),.36,label='Within RAW CV',color='#2878B5')
        left.bar(x+.18,values('across_mean_CV_percent'),.36,label='Across independent effective frames CV',color='#F28E2B')
        left.set_xticks(ticks,[f'R{k}' for k in ticks],rotation=45);left.set_ylabel('CV (%)');left.legend(fontsize=8);left.grid(axis='y',alpha=.2)
        matrix=np.array([[np.nan if v is None else v for v in row] for row in info['correlation']])
        im=right.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=right)
        right.set_xticks(ticks,[f'R{k}' for k in ticks],rotation=45);right.set_yticks(ticks,[f'R{k}' for k in ticks]);right.set_title('Across-visit correlation (descriptive)')
        fig.suptitle(f'{name} | RAW {result["settings"]["filter_mode"]} | N={result["settings"]["frames"]} | alpha={result["settings"]["filter_alpha"]} | independent windows')
        for ext in ('png','pdf'):fig.savefig(out/f'diagnostics_{name}.{ext}',dpi=160)
    names=list(result['inputs']);matrix=np.full((len(names),len(names)),np.nan)
    for p in result['separation_pairs']:
        if p['separation_ratio'] is not None:
            i,j=names.index(p['a']),names.index(p['b']);matrix[i,j]=matrix[j,i]=p['separation_ratio']
    fig=Figure(figsize=(8,6),constrained_layout=True);FigureCanvasAgg(fig);ax=fig.subplots()
    im=ax.imshow(matrix,vmin=0,vmax=max(3.,float(np.nanmax(matrix))) if np.isfinite(matrix).any() else 3.,cmap='viridis')
    fig.colorbar(im,ax=ax,label='Mean distance / across-visit scatter (screen >= 3)')
    ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white',fontsize=8)
    ax.set_title('Input separation | RAW | not a control-accuracy test')
    for ext in ('png','pdf'):fig.savefig(out/f'separation.{ext}',dpi=160)
    for filename,rows in [('readiness.csv',result['readiness']['gates']),('separation.csv',result['separation_pairs']),
                          ('noise_drift.csv',[dict(input=g,**r) for g,v in result['diagnostics'].items() for r in v['regions']])]:
        with (out/filename).open('x',encoding='utf-8-sig',newline='') as f:
            if rows:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# 读出稳定性与部署前筛查','',result['readiness']['recommendation'],'',
          f'主读出为RAW有效帧：{result["settings"]["filter_mode"]}，N={result["settings"]["frames"]}，α={result["settings"]["filter_alpha"]}。每窗独立清空；与主界面同算法。显示参数另做处理后对照，无参考帧或补偿。',
          f'本轮{result["region_count"]}区。封存RAW {result["accepted_raw_total"]}张；各输入独立统计，不把N张原始帧当N次独立实验。','',
          '| 项目 | 判定 | 本轮数值/缺项 | 门槛/解释 |','|---|---|---|---|']
    for r in result['readiness']['gates']:
        text.append('| '+' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ('item','status','value','criterion'))+' |')
    text += ['',f'覆盖的最大绝对X/Y/VX/VY：{result["coverage"]["max_abs_xy_vxy"]}（m、m/s）。',
             '时间跨度、趋势、区间前后变化和相关性是描述性诊断，不证明物理噪声原因或长期稳定。',
             '完整数值见RESULT.json、regions.csv、noise_drift.csv；图表同时保存PNG/PDF，原始RAW不改变。']
    (out/'总结.md').write_text('\n'.join(text),encoding='utf-8')
