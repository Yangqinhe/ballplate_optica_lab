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
    reference_frames: int = 0
    baseline_frames: int = 50

    def validate(self):
        if type(self.reference_frames) is not int or self.reference_frames < 0:
            raise ValueError('参考帧预算必须为非负整数')
        if self.reference_frames and not 4 <= self.reference_frames <= self.frames-2:
            raise ValueError('参考总帧数至少4，且必须给目标保留至少2张；N包含参考和目标')
        if type(self.baseline_frames) is not int or not 2 <= self.baseline_frames <= 120:
            raise ValueError('独立参考基线需要2–120张RAW，额外计数')
        if type(self.frames) is not int or not 2 <= self.frames <= 120:
            raise ValueError('每个有效帧必须使用2–120张RAW')
        if type(self.repeats) is not int or not 3 <= self.repeats <= 20:
            raise ValueError('每种输入至少3次、最多20次独立回访')
        if type(self.switch_inputs) is not bool:
            raise ValueError('换图必须为明确布尔选项')
        for name, low, high in (('settle_seconds', 1.5, 15), ('interval_seconds', 0, 15),
                               ('software_gain', 1, 32), ('black_ADU', 0, 254),
                               ('gamma', .1, 3), ('maximum_seconds', 5, 300)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('参数超出范围：'+name)
        return self


def schedule(settings):
    settings.validate()
    if settings.reference_frames:
        result = [dict(input_id='R', visit=0, role='baseline', frames=settings.baseline_frames)]
        for i in range(settings.repeats):
            for label in (('A', 'B') if settings.switch_inputs else ('A',)):
                for role, source, count in [('pre','R',settings.reference_frames//2),
                        ('target',label,settings.frames-settings.reference_frames),
                        ('post','R',settings.reference_frames-settings.reference_frames//2)]:
                    result.append(dict(input_id=source, visit=i+1, role=role, target=label, frames=count))
        return result
    return [dict(input_id=label, visit=index+1)
            for index in range(settings.repeats)
            for label in (('A', 'B') if settings.switch_inputs else ('A',))]


def mapping(raw, settings):
    """Map after RAW averaging, identically for preview-of-result and readout."""
    settings.validate()
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any() or (a > 255).any():
        raise ValueError('finite native Mono8-valued plane required')
    unit = np.clip((a-settings.black_ADU)*settings.software_gain/255., 0, 1)
    return np.rint(255.*unit**settings.gamma).astype(np.uint8)


def validate_weights(weights, shape):
    w = np.asarray(weights)
    if w.shape != (8, *shape) or not np.isin(w, (0, 1)).all() or not np.all(w.sum(0) == 1):
        raise ValueError('八区必须为对应当前ROI、完整无重叠的固定二值mask')
    if np.any(w.sum((1, 2)) == 0):
        raise ValueError('八区不能包含空区域')
    return w


def readout(raws, weights, settings):
    settings.validate()
    a = np.asarray(raws)
    if a.ndim != 3 or a.dtype != np.uint8 or len(a) != settings.frames:
        raise ValueError('one complete independent N-frame native uint8 batch required')
    w = validate_weights(weights, a.shape[1:])
    mean = a.astype(np.float32).mean(0, dtype=np.float32)
    processed = mapping(mean, settings)
    raw8 = np.einsum('hw,khw->k', mean.astype(np.float64), w, optimize=True)
    processed8 = np.einsum('hw,khw->k', processed.astype(np.float64), w, optimize=True)
    # This within-window statistic is RAW, not the variance of independent means.
    labels = w.argmax(0).ravel()
    single = np.array([np.bincount(labels, weights=frame.ravel(), minlength=8) for frame in a])
    regions = []; any_clipped = (a == 255).any(0)
    for k in range(8):
        values = a[:, w[k].astype(bool)]
        clip = values == 255; near = values >= 240
        yy, xx = np.where(any_clipped & w[k].astype(bool))
        regions.append(dict(region=k, max_ADU=int(values.max()),
            clipped_pixel_observations=int(clip.sum()), pixel_observations=int(values.size),
            clipped_percent=100*float(clip.mean()),
            max_frame_clipped_percent=100*float(clip.mean(1).max()),
            unique_clipped_pixels=int(clip.any(0).sum()),
            near_saturation_pixel_observations=int(near.sum()),
            near_saturation_percent=100*float(near.mean()),
            clipped_bbox_xyxy=([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None)))
    return dict(raw8=raw8.tolist(), processed8=processed8.tolist(), brightness_regions=regions,
                brightness_warning=bool(a.max()>=240),
                within_window_RAW_SD8=single.std(0, ddof=1).tolist(),
                max_ADU=int(a.max()), clipped_pixels=int(np.count_nonzero(a == 255)),
                frames=len(a)), mean, processed


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
    if x.ndim != 2 or x.shape[1] != 8 or len(x) < 3 or not np.isfinite(x).all() or (not allow_signed and (x < 0).any()):
        raise ValueError('每种输入至少三次独立、非负八区读数')
    rows = [dict(region=k, **column_summary(x[:, k])) for k in range(8)]
    mean = x.mean(0); norm = float(np.linalg.norm(mean))
    rms = float(np.sqrt(np.mean(np.sum((x-mean)**2, axis=1))))
    pair = max(float(np.linalg.norm(a-b)) for i, a in enumerate(x) for b in x[i+1:])
    s = x[:, 4]+x[:, 5]; d = x[:, 4]-x[:, 5]
    correlation = None
    if x[:, 4].std() > 0 and x[:, 5].std() > 0:
        correlation = float(np.corrcoef(x[:, 4], x[:, 5])[0, 1])
    normalized = ((d/s).tolist() if np.all(s > 1) else None)
    vector_rms = 100*rms/norm if norm > 1 else None
    maximum_pair = 100*pair/norm if norm > 1 else None
    cvs = [r['CV_percent'] for r in rows]
    gate = ('UNKNOWN' if vector_rms is None or any(c is None for c in cvs) else
            'RESEARCH_PASS' if vector_rms <= 1 and maximum_pair <= 3 and max(cvs) <= 3 else 'RESEARCH_FAIL')
    return dict(independent_visits=len(x), regions=rows, relative_vector_RMS_percent=vector_rms,
                maximum_pair_L2_percent=maximum_pair, maximum_region_CV_percent=(max(cvs) if all(c is not None for c in cvs) else None),
                reference_gate=gate, gate_definition='RMS≤1%; max pair≤3%; each CV≤3%; descriptive research only',
                pair45=dict(sum=column_summary(s), difference_mean=float(d.mean()),
                            difference_SD=float(d.std(ddof=1)), difference_range=float(np.ptp(d)),
                            correlation=correlation, normalized_difference=normalized,
                            signed_difference_CV_NOT_used=True),
                control_qualified=False, physical_cause_identified=False)


def analyze(visits, settings):
    expected = schedule(settings)
    if len(visits) != len(expected) or any(any(v[k] != e[k] for k in ('input_id', 'visit')) for v, e in zip(visits, expected)):
        raise ValueError('完整预声明顺序未完成；失败前缀不能生成正式波动判定')
    output = {}; photometry = {}
    for label in ('A', 'B') if settings.switch_inputs else ('A',):
        selected = [v for v in visits if v['input_id'] == label]
        output[label] = {domain: summary([v[domain+'8'] for v in selected]) for domain in ('raw', 'processed')}
        complete = all('brightness_regions' in v for v in selected)
        clipped = any(v.get('clipped_pixels', 0)>0 for v in selected)
        near = any(v.get('max_ADU', 0)>=240 for v in selected)
        regions = []
        if complete:
            for k in range(8):
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
    return dict(schema='optical-lab-stability-results-1', settings=asdict(settings), inputs=output,
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
    raws = []; stamps = []; ids = []; host_stamps = []
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
        raws.append(raw); stamps.append(int(meta['original_metadata']['TimeStamp']))
        ids.append(int(meta['original_metadata']['FrameID']))
        host_stamps.append(meta['original_metadata'].get('host_received_unix_ns'))
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('CCD时间戳未严格递增')
    row, mean, processed = readout(np.stack(raws), weights, settings)
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
    x = np.asarray([v[domain+'8'] for v in selected], dtype=np.float64)
    stats = summary(x, allow_signed=True)
    means = x.mean(0); valid = means > 1
    deviations = np.full_like(x, np.nan)
    deviations[:, valid] = 100*(x[:, valid]/means[valid]-1)
    extent = max(1., float(np.nanmax(np.abs(deviations)))*1.15) if valid.any() else 1.
    fig = Figure(figsize=(12, 5.5), constrained_layout=True); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4, sharex=True, sharey=True)
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
    count = selected[0].get('frames',result['settings']['frames'])
    geometry=result.get('input_geometry',{}).get(label,{})
    size=f' | ball dilation R={geometry["radius_raw_px"]} RAW96 px' if geometry.get('kind')=='four_ball_history' else ''
    fig.suptitle(f'Input {label} | {domain} | {count} target RAW / visit | all 8 regions'+size)
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
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 7), constrained_layout=True); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for label, domains in result['inputs'].items():
        selected = [v for v in result['visits'] if v['input_id'] == label]
        x = np.array([v['processed8'] for v in selected], float)
        cv = [np.nan if r['CV_percent'] is None else r['CV_percent'] for r in domains['processed']['regions']]
        axes[0, 0].plot(range(8), cv, 'o-', label=label)
        indices = range(1, len(x)+1)
        for k, color in ((4, '#d88b10'), (5, '#cc39a8')):
            axes[0, 1].plot(indices, x[:, k], '-' if label == 'A' else '--', color=color, marker='o', label=f'{label}: R{k}')
        axes[1, 0].plot(indices, x[:, 4]+x[:, 5], 'o-', label=label)
        axes[1, 1].plot(indices, x[:, 4]-x[:, 5], 'o-', label=label)
    for ax, title in zip(axes.flat, ['Across-visit CV (%)', 'R4 / R5', 'R4 + R5', 'R4 - R5']):
        ax.set_title(title); ax.grid(alpha=.2); ax.legend(); ax.set_xlabel('Region' if ax is axes[0, 0] else 'Independent visit')
    axes[0, 0].axhline(3, color='gray', linestyle=':', label='3% reference')
    fig.suptitle(f'Fixed-input repeatability | N={result["settings"]["frames"]} | research only')
    for ext in ('png', 'pdf'): fig.savefig(out/('stability_diagnostics.'+ext), dpi=160)
    # Primary GUI output: all eight traces, selected by the same A/B and domain
    # controls as the table. Retain the R4/R5 diagnostics as secondary files.
    for label in result['inputs']:
        for domain in result['inputs'][label]:
            chart = region_figure(result, label, domain)
            for ext in ('png', 'pdf'):
                chart.savefig(out/(f'stability_{label}_{domain}.'+ext), dpi=160)
                if label == 'A' and domain == 'processed':
                    chart.savefig(out/('stability.'+ext), dpi=160)
    text = ['# 八区稳定性测试', '', '每次有效帧：独立N张RAW均值，再黑电平/γ处理。A/B分别统计，不混合。',
            '指标不代表光子SNR、物理根因、全天稳定或闭环资格。',
            '主图同时展示R0–R7相对各自均值的波动，八区共用纵轴尺度；A/B及RAW/处理后分别出图。R4/R5和差诊断保留在stability_diagnostics文件。',
            '亮度告警不终止采样或统计；含255ADU数据标记SATURATED_UNQUALIFIED，即使数值CV为零也不能用于证明部署达标。各区亮度见brightness_regions.csv，逐窗像素位置见MEAN目录的clipped_hits.npy。', '']
    for label, domains in result['inputs'].items():
        d = domains['processed']; text += [f'## 输入{label}', '',
            f'独立回访{d["independent_visits"]}次；相对向量RMS={d["relative_vector_RMS_percent"]}%；研究参考判定={d["reference_gate"]}。', '',
            '| Region | 均值 | 样本SD | CV% | 峰峰差/均值% |', '|---|---:|---:|---:|---:|']
        for r in d['regions']:
            text.append(f'| {r["region"]} | {r["mean"]:.4f} | {r["sample_SD"]:.4f} | {r["CV_percent"]} | {r["peak_to_peak_percent"]} |')
        text += ['', 'R4/R5差分可接近零，不以差分CV衡量稳定性；详见JSON中的差分SD、范围和归一化差分。', '']
    (out/'总结.md').write_text('\n'.join(text), encoding='utf-8')
    if 'comparison' in result:
        export_comparison(result, out)
    hashes = {p.name: file_ref(p)['sha256'] for p in sorted(out.iterdir()) if p.is_file()}
    with (out/'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(hashes, f, indent=2)
    return out


CONTROL_LIMITS = {
    'MAE_rad': .06, 'p95_error_L2_rad': .18, 'center_max_L2_rad': .05,
    'center_max_pair_rad': .04, 'same_input_pair_p95_rad': .04,
    'same_input_pair_max_rad': .08,
}


def load_head(path):
    """Explicit file-only selection; no training or control execution."""
    ref = file_ref(path); head = json.loads(Path(ref['path']).read_text())
    w = np.asarray(head.get('weights_8x2'), float); b = np.asarray(head.get('bias_2'), float)
    if (head.get('parameter_count') != 18 or w.shape != (8, 2) or b.shape != (2,)
            or not np.isfinite(w).all() or not np.isfinite(b).all()
            or 'RAW' not in head.get('feature_domain', '') or 'fixed_camera' not in head
            or not head.get('optical_mask_sha256') or not head.get('readout_mask')):
        raise ValueError('需要显式18参数RAW线性头、八区mask、SLM及相机配置元数据')
    return dict(reference=ref, head=head)


def fit_affine(reference, baseline):
    x = np.asarray(baseline, float).ravel(); y = np.asarray(reference, float).ravel()
    xc = x-x.mean(); variance = float(xc@xc)
    if variance/len(x) < 1e-4: raise ValueError('参考基线空间方差不足1e-4 ADU²，不能可靠拟合比例')
    gain = float(xc@(y-y.mean())/variance); offset = float(y.mean()-gain*x.mean())
    if not .8 <= gain <= 1.2:
        raise ValueError('参考比例超出预声明0.8–1.2范围；不夹紧或伪造补偿')
    return gain, offset


def mean_plane(row):
    ref = row['mean_file']; path = Path(ref['path'])
    if file_ref(path) != ref: raise ValueError('有效帧SHA改变')
    return np.load(path, allow_pickle=False).astype(np.float64)


def assess_controls(visits, domains, head_record, labels, context):
    """Snapshot head only; labels score outputs, never modify those outputs."""
    if head_record is None:
        return dict(status='UNKNOWN_NO_HEAD', domains={}, deployment_qualified=False)
    head = head_record['head']; w = np.asarray(head['weights_8x2']); b = np.asarray(head['bias_2'])
    mismatches = []
    if head['readout_mask']['sha256'] != context['mask_sha256']: mismatches.append('八区mask不同')
    if head['optical_mask_sha256'] != context['slm_sha256']: mismatches.append('SLM权重不同')
    for k, v in head['fixed_camera'].items():
        if context['camera_actual'].get(k) != v: mismatches.append('相机 '+k)
    if any(v['frames'] != 50 for v in visits): mismatches.append('目标均值帧数不等于头训练时N50')
    for name,geometry in (context.get('input_geometry') or {}).items():
        if name=='R':continue
        if geometry.get('kind')=='four_ball_history' and not geometry.get('matches_original_R2_bytes',False):
            mismatches.append(f'输入{name}球尺寸R={geometry.get("radius_raw_px")}，不同于原R2编码；需重新适配，当前只诊断')
        elif geometry.get('kind')=='unknown':mismatches.append(f'输入{name}球尺寸未知，不能继承旧头资格')
    result = {}
    for domain in domains:
        values = np.asarray([v[domain+'8'] for v in visits]); actions = values@w+b
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


def analyze_comparison(visits, settings, weights, head_record=None, labels=None, context=None):
    """File worker: bounded planes in memory, RAW always retained and untouched."""
    expected = schedule(settings)
    if len(visits)!=len(expected) or any(any(v.get(k)!=e[k] for k in e) for v,e in zip(visits,expected)):
        raise ValueError('完整预声明序列未完成，不能用失败前缀给出判定')
    rows = [dict(v) for v in visits if v.get('role','target')=='target']
    targets = [dict(v) for v in rows]
    plain = replace(settings, reference_frames=0)
    result = analyze(targets, plain)
    result['settings'] = asdict(settings)
    result['input_geometry'] = (context or {}).get('input_geometry',{})
    reference_info = []; errors = []
    if settings.reference_frames:
        base = mean_plane(visits[0]); base8 = np.einsum('hw,khw->k',base,weights)
        for index, (pre, target, post) in enumerate(zip(visits[1::3],visits[2::3],visits[3::3])):
            try:
                ts = [v.get('mean_host_received_ns') for v in (pre,target,post)]
                if any(t is None for t in ts) or not ts[0]<ts[1]<ts[2]:
                    raise ValueError('缺少严格先后顺序的真实采样时间戳')
                fraction = (ts[1]-ts[0])/(ts[2]-ts[0])
                before = mean_plane(pre); after = mean_plane(post)
                gp,bp = fit_affine(before,base); gq,bq = fit_affine(after,base)
                g = (1-fraction)*gp+fraction*gq; b = (1-fraction)*bp+fraction*bq
                raw = np.asarray(target['raw8']); r = (1-fraction)*np.asarray(pre['raw8'])+fraction*np.asarray(post['raw8'])
                rows[index]['additive8'] = (raw-r+base8).tolist()
                rows[index]['affine8'] = ((raw-b*weights.sum((1,2)))/g).tolist()
                reference_info.append(dict(target=target['input_id'],visit=target['visit'],gain=g,offset_ADU=b,
                    pre_gain=gp,post_gain=gq,interpolation_fraction=fraction,
                    pre_gap_s=(ts[1]-ts[0])/1e9,post_gap_s=(ts[2]-ts[1])/1e9))
            except ValueError as error:
                errors.append(dict(target=target['input_id'],visit=target['visit'],error=str(error)))
        if not errors:
            for label in result['inputs']:
                for domain in ('additive','affine'):
                    stats = summary([v[domain+'8'] for v in rows if v['input_id']==label],allow_signed=True)
                    if any(v.get('clipped_pixels',0) for v in visits): stats['reference_gate']='SATURATED_UNQUALIFIED'
                    result['inputs'][label][domain] = stats
        else:
            # No cherry-picking: preserve partial fits, but no aggregate compensated score.
            for row in rows:
                row.pop('additive8',None);row.pop('affine8',None)
    result['visits']=rows
    domains=list(next(iter(result['inputs'].values())))
    result['comparison'] = dict(reference_enabled=bool(settings.reference_frames),
        accepted_raw_total=sum(v['frames'] for v in visits),target_frames_per_effective=rows[0]['frames'],
        baseline_frames_extra=settings.baseline_frames if settings.reference_frames else 0,
        reference_frames_per_effective=settings.reference_frames,reference_fit=reference_info,reference_errors=errors,
        acquisition_visits=visits,reference_never_mixed_into_target=True,each_query_history_reset=True,
        baseline_shared_uncertainty_not_independent=True if settings.reference_frames else False,
        reference_schedule='R0(extra) then Rpre/T/Rpost; not per-frame hardware interleaving',
        reference_saturation=any(v.get('clipped_pixels',0)>0 for v in visits if v.get('role')!='target') if settings.reference_frames else False)
    result['control_analysis']=assess_controls(rows,domains,head_record,labels or {},context or {})
    return result


def export_comparison(result, out):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows=result['visits']; controls=result['control_analysis']; comparison=result['comparison']
    with (out/'input_geometry.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['input','kind','radius_RAW96_px','frame','bbox_RAW96_wh','bbox_DMD_wh','DMD_on_pixels','centroid_delta_RAW96_px'])
        for label,geometry in result.get('input_geometry',{}).items():
            for item in geometry.get('geometry',[{}]):
                writer.writerow([label,geometry.get('kind'),geometry.get('radius_raw_px'),item.get('frame_index'),
                    item.get('encoded_bbox_wh_raw_px'),item.get('encoded_bbox_wh_DMD_px'),item.get('DMD_on_pixels'),item.get('centroid_delta_L2_raw_px')])
    with (out/'control_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['domain','metric','value_rad','limit_rad','local_status','context'])
        for domain,info in controls['domains'].items():
            for name,g in info['gates'].items():writer.writerow([domain,name,g['value'],g['limit'],g['status'],info['compatibility']])
    with (out/'effective_readouts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['input','visit','domain','target_RAW_count',*[f'R{k}' for k in range(8)]])
        for v in rows:
            for domain in result['inputs'][v['input_id']]:writer.writerow([v['input_id'],v['visit'],domain,v['frames'],*v[domain+'8']])
    colors={'raw':'#2878B5','processed':'#888888','additive':'#E07B39','affine':'#369A70'}
    for label,domains in result['inputs'].items():
        fig=Figure(figsize=(12,5.5),constrained_layout=True);FigureCanvasAgg(fig);axes=fig.subplots(2,4,sharex=True,sharey=True)
        selected=[v for v in rows if v['input_id']==label];extent=1.
        for domain in domains:
            x=np.asarray([v[domain+'8'] for v in selected]);m=x.mean(0);ok=np.abs(m)>1
            d=np.full_like(x,np.nan);d[:,ok]=100*(x[:,ok]-m[ok])/np.abs(m[ok])
            if ok.any():extent=max(extent,float(np.nanmax(np.abs(d)))*1.1)
            for k,ax in enumerate(axes.flat):ax.plot(range(1,len(x)+1),d[:,k],'o-',color=colors[domain],label=domain,ms=3)
        for k,ax in enumerate(axes.flat):
            ax.set_title(f'R{k}');ax.set_ylim(-extent,extent);ax.axhline(0,c='gray',lw=.5);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8);fig.supxlabel('Independent target revisit');fig.supylabel('Deviation from domain mean (%)')
        fig.suptitle(f'Input {label} | paired readout comparison | target RAW={selected[0]["frames"]}')
        for ext in ('png','pdf'):fig.savefig(out/f'comparison_{label}.{ext}',dpi=160)
    fig=Figure(figsize=(12,7),constrained_layout=True);FigureCanvasAgg(fig);axes=fig.subplots(2,2)
    for domain,info in controls['domains'].items():
        a=np.asarray(info['actions_rad']);x=np.arange(1,len(a)+1)
        for k in range(2):axes[0,k].plot(x,a[:,k],'o-',color=colors[domain],label=domain,ms=3)
        values=[np.nan if info['metrics'][key] is None else info['metrics'][key]/CONTROL_LIMITS[key] for key in CONTROL_LIMITS]
        axes[1,0].plot(range(6),values,'o-',color=colors[domain],label=domain)
        sensitivity=np.linalg.norm(np.asarray(info['region_action_RMS_rad']),axis=1)
        axes[1,1].plot(range(8),sensitivity,'o-',color=colors[domain],label=domain)
    first=next(iter(controls['domains'].values()),None)
    if first:
        for k in range(2):
            teacher=[np.nan if v is None else v[k] for v in first['teacher_actions_rad']]
            axes[0,k].plot(range(1,len(teacher)+1),teacher,'k--',label='teacher (score only)')
    else:
        axes[0,0].text(.5,.5,'UNKNOWN: no head selected',ha='center',transform=axes[0,0].transAxes)
    for ax,title in zip(axes.flat,['Roll output (rad)','Pitch output (rad)','Local metric / engineering limit','Per-region output fluctuation contribution (rad)']):
        ax.set_title(title);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    axes[1,0].axhline(1,color='red',ls='--');axes[1,0].set_xticks(range(6),['MAE','p95 err','center','ctr pair','pair p95','pair max'],rotation=20)
    axes[1,1].set_xticks(range(8),[f'R{k}' for k in range(8)])
    fig.suptitle('Fixed 18-parameter head | NO refit / NO action clipping / NOT deployment qualification')
    for ext in ('png','pdf'):fig.savefig(out/f'control_comparison.{ext}',dpi=160)
    for domain,info in controls['domains'].items():
        chart=Figure(figsize=(12,6),constrained_layout=True);FigureCanvasAgg(chart);panels=chart.subplots(2,2)
        a=np.asarray(info['actions_rad']);x=np.arange(1,len(a)+1)
        for k in range(2):
            panels[0,k].plot(x,a[:,k],'o-',color=colors[domain],label=domain,ms=4)
            t=[np.nan if v is None else v[k] for v in info['teacher_actions_rad']]
            panels[0,k].plot(x,t,'k--',label='teacher (score only)')
            panels[0,k].set_title(('Roll','Pitch')[k]+' output (rad)');panels[0,k].set_xlabel('Target revisit (A/B order)')
        vals=[np.nan if info['metrics'][k] is None else info['metrics'][k]/CONTROL_LIMITS[k] for k in CONTROL_LIMITS]
        panels[1,0].bar(range(6),vals,color=colors[domain]);panels[1,0].axhline(1,c='red',ls='--')
        panels[1,0].set_xticks(range(6),['MAE','p95 err','center','ctr pair','pair p95','pair max'],rotation=20)
        panels[1,0].set_title('Local metric / engineering limit (missing = UNKNOWN)')
        contribution=np.asarray(info['region_action_RMS_rad'])
        for k in range(2):panels[1,1].plot(range(8),contribution[:,k],'o-',label=('roll','pitch')[k])
        panels[1,1].set_xticks(range(8),[f'R{k}' for k in range(8)]);panels[1,1].set_title('Per-region output RMS contribution (rad)')
        for ax in panels.flat:
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        chart.suptitle(domain+' | fixed 18-parameter head | STATIC DIAGNOSTIC ONLY')
        for ext in ('png','pdf'):chart.savefig(out/f'control_{domain}.{ext}',dpi=160)
    lines=['# 参考补偿与控制敏感性对照','',f'封存RAW总数：{comparison["accepted_raw_total"]}；每个目标均值{comparison["target_frames_per_effective"]}张。',
        f'每个有效帧另用参考{comparison["reference_frames_per_effective"]}张；独立起始基线额外{comparison["baseline_frames_extra"]}张。',
        '参考不混入目标；每个采样窗清空历史；共享起始基线的误差可能相关。此模式是分块R→T→R，非逐帧硬同步。',
        '所有路线基于同一目标RAW，参考补偿不等于增加目标帧数。普通N50与5+40+5不是同一目标帧数的实验。',
        '六项阈值是工程研究门槛，并非稳定性定理。只测A/B不能证明全位置、速度、长时间或MuJoCo闭环达标。',
        '显示软件增益/黑电平/γ不会写入相机；processed接RAW头仅用于敏感性展示，不是适配后控制。',
        f'参考拟合失败（任一失败则不输出整组补偿统计）：{comparison["reference_errors"]}',
        f'参考截顶：{comparison["reference_saturation"]}；截顶继续记录，但不提供部署证明。','',
        '| 路线 | 指标 | 实测rad | 参考上限rad | 子集数值判定 |','|---|---|---:|---:|---|']
    for domain,info in controls['domains'].items():
        for name,g in info['gates'].items():lines.append(f'| {domain} | {name} | {g["value"]} | {g["limit"]} | {g["status"]} |')
        lines+=['',f'{domain} 上下文提醒：{info["context_warnings"]}','']
    if not controls['domains']:lines.append('未选控制头：控制指标UNKNOWN；八区分析仍有效。')
    lines+=['','## 本轮输入球尺寸（冻结值，不是界面当前值）','']
    for label,geometry in result.get('input_geometry',{}).items():
        if geometry.get('kind')=='four_ball_history':
            lines.append(f'- {label}：R={geometry["radius_raw_px"]}（96图扩张半径）；DMD总亮像素{geometry["DMD_on_pixels"]}；'
                         +f'四球DMD包围盒{[v["encoded_bbox_wh_DMD_px"] for v in geometry["geometry"]]}。')
        else:lines.append(f'- {label}：{geometry.get("kind","unknown")}，不推断球大小。')
    lines+=['','R0为原始小球，R2为旧编码；只有球轮廓变化，位置、历史顺序、3倍复制和12像素边框不变。尺寸改变需重新评估控制映射，旧头仅诊断。']
    if result.get('historical_replay_note'):lines[2:2]=['**历史数据回放，非新实测：** '+result['historical_replay_note'],'']
    (out/'对照总结.md').write_text('\n'.join(lines),encoding='utf-8')
