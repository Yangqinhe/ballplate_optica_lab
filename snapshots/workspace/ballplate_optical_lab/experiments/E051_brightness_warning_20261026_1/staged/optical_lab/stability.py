"""Fixed-input repeatability: pure planning/math; explicit file-only export.

No Qt, device SDK, process discovery or hardware action on import. Independent
visits, not constituent RAW frames, are the samples for the across-visit table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def validate(self):
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


def summary(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8 or len(x) < 3 or not np.isfinite(x).all() or (x < 0).any():
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
    raws = []; stamps = []; ids = []
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
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('CCD时间戳未严格递增')
    row, mean, processed = readout(np.stack(raws), weights, settings)
    # TimeStamp is in ns in this camera adapter, not a host preview timestamp.
    duration = (stamps[-1]-stamps[0])/1e9
    row.update(item, archive=reference, sample_span_seconds=duration,
               observed_rate_hz=(settings.frames-1)/duration if duration > 0 else None,
               first_frame_id=ids[0], last_frame_id=ids[-1],
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
    if label not in result['inputs'] or domain not in ('raw', 'processed'):
        raise ValueError('explicit result input/domain required')
    selected = [v for v in result['visits'] if v['input_id'] == label]
    x = np.asarray([v[domain+'8'] for v in selected], dtype=np.float64)
    stats = summary(x)
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
    fig.suptitle(f'Input {label} | {domain} | N={result["settings"]["frames"]} RAW / visit | all 8 regions')
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
        for domain in ('raw', 'processed'):
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
    hashes = {p.name: file_ref(p)['sha256'] for p in sorted(out.iterdir()) if p.is_file()}
    with (out/'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(hashes, f, indent=2)
    return out
