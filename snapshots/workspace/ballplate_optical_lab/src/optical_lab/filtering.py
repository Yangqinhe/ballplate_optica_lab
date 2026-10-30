"""Pure CCD filters: display-only temporal preview and historical assessment.

No import opens Qt, SDK or a camera. Archived E005 spatial-filter measurements
remain historical evidence, never validation of a subsequently realigned setup.
"""
from pathlib import Path
import base64
import hashlib
import html
import io
import json
import numbers

import numpy as np

from . import observability_assessment as obsa
from .protocol import observability_states
from .reporting import plotting, save_figure
from .storage import sha256, write_json_new

ROI = obsa.ROI
BROAD8_SHA256 = obsa.BROAD8_SHA256
SIGMAS = (0., .5, 1., 2.)
GROUP_ORDER = (("learned",0,"POS"),("learned",2,"VEL"),("learned",2,"POS"),("learned",0,"VEL"),
               ("flat",2,"POS"),("flat",2,"VEL"),("lens",2,"VEL"),("lens",2,"POS"))


class PreviewTemporalFilter:
    """RAW preview only: sampled GUI frames, NOT all camera frames or actor input.

    Mean precedes display gamma. status is an independent dict; count means
    contributing samples (off1 / mean2<=2 / EMA since reset). Same-key repeats
    within unchanged context/config do not update value, sample count or time.
    """
    def __init__(self):
        self.reset(); self._reason = 'initial'

    def reset(self):
        self._signature = self._key = self._time = self._previous = self._value = None
        self._observed = 0; self._reason = 'manual'; self._interval = None; self._duplicate = False

    @property
    def status(self):
        mode = self._signature[0] if self._signature is not None else 'off'
        count = min(self._observed, 2 if mode == 'mean2' else 1) if mode != 'ema' else self._observed
        return dict(mode=mode, count=count, observed_count=self._observed, reset_reason=self._reason,
                    frame_interval_ms=self._interval, duplicate=self._duplicate, preview_sampled_only=True)

    def update(self, raw, *, frame_key, context, mode='off', alpha=.5, now=None):
        import time
        value = np.asarray(raw)
        if value.dtype != np.uint8 or value.ndim != 2 or value.size == 0 or value.size > 1048576:
            raise ValueError('nonempty uint8 HxW RAW, at most1048576 pixels, required')
        if mode not in ('off', 'mean2', 'ema') or isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('off/mean2/ema and finite nonboolean alpha in (0,1] required')
        key = frame_key if isinstance(frame_key, tuple) else (frame_key,)
        if not key or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0 for k in key):
            raise ValueError('nonnegative integer frame key or tuple of integers required')
        key = tuple(int(k) for k in key); stamp = time.monotonic() if now is None else now
        if isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, numbers.Real) or not np.isfinite(stamp) or stamp < 0:
            raise ValueError('finite nonnegative monotonic preview sample time required')
        try: frozen = json.dumps(context, sort_keys=True, allow_nan=False, separators=(',', ':'))
        except (TypeError, ValueError) as error: raise ValueError('finite JSON-serializable context required') from error
        signature = (mode, float(alpha), value.shape, frozen); reason = None
        if self._signature is not None and signature != self._signature:
            reason = next(name for name, a, b in zip(('mode_changed', 'alpha_changed', 'shape_changed', 'context_changed'), signature, self._signature) if a != b)
        if reason is None and key == self._key:
            self._duplicate = True; return self._value.copy()
        if reason is None and self._key is not None:
            if len(key) != len(self._key): reason = 'key_shape_changed'
            elif key < self._key: reason = 'frame_order_reversed'
            elif stamp < self._time: reason = 'clock_reversed'
            elif stamp-self._time > .5: reason = 'sample_gap_gt_500ms'
        if reason is not None:
            self.reset(); self._reason = reason
        interval = None if self._time is None else (float(stamp)-self._time)*1000.
        current = value.astype(np.float32, copy=True)
        if self._value is None or mode == 'off': filtered = current.copy()
        elif mode == 'mean2': filtered = (self._previous+current)*np.float32(.5)
        else: filtered = self._value+np.float32(alpha)*(current-self._value)
        self._value = filtered; self._previous = current; self._signature = signature
        self._key = key; self._time = float(stamp); self._observed += 1; self._interval = interval; self._duplicate = False
        return filtered.copy()


def spatial_lowpass(array, sigma):
    """Independent float64 output; Gaussian only on H/W, reflect/truncate=3."""
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, numbers.Real) or not np.isfinite(sigma) or not 0 <= sigma <= 2:
        raise ValueError("sigma must be a finite real scalar in [0,2], not bool/string/list")
    value = np.asarray(array)
    if (value.ndim not in (2,3) or any(n == 0 for n in value.shape) or value.dtype.kind not in "uif"
            or not np.isfinite(value).all()):
        raise ValueError("nonempty finite real numeric 2D or (N,H,W) input required")
    result = value.astype(np.float64, copy=True)
    if sigma == 0:
        return result
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(result, sigma=(float(sigma),)*2 if result.ndim == 2 else (0.,float(sigma),float(sigma)),
                           mode="reflect", truncate=3., output=np.float64)


def _ratio(a, b):
    return float(a/b) if b > 0 else None


def _median(values):
    defined = [v for v in values if v is not None]
    return float(np.median(defined)) if defined else None


def _features(stack, labels):
    return np.stack([np.bincount(labels, weights=frame.ravel(), minlength=8) for frame in stack])


def _load_capture(reader, ref, index, seen):
    root, _ = reader.archive(ref["path"], ref["sha256"])
    plan = reader.js(root/"PLAN.json"); result = reader.js(root/"RESULT.json")
    spec = plan["batch_spec"]; expected = observability_states(spec["axis_family"])
    if (plan.get("profile") != "observability_v1" or plan.get("batch_id") != ref["batch_id"]
            or result.get("batch_id") != ref["batch_id"] or plan.get("observability_bank") != index["bank"]
            or plan.get("states") != expected or plan.get("frames_per_state") != 8 or plan.get("total_planned_frames") != 88
            or result.get("complete") is not True or result.get("captured_valid_frames") != 88
            or any(result.get(k) is not True for k in ("actual_outputs_zero","watchdog_verified","camera_left_running","frozen_inputs_unchanged"))
            or any(result.get(k) for k in ("error","cleanup_error","archive_error","integrity_error","stopped_by_bound","resource_cleanup_errors"))):
        raise ValueError("complete sealed 11x8 historical capture/profile required")
    if plan["camera_expected"]["roi"] != ROI or plan["regions"]["sha256"] != BROAD8_SHA256:
        raise ValueError("fixed original native ROI/BROAD8 required")
    regions = reader.npy(plan["regions"]["path"], BROAD8_SHA256)
    shape = (ROI["Height"],ROI["Width"])
    if regions.shape != (8,*shape) or not np.isin(regions,(0,1)).all() or not np.all(regions.sum(0) == 1):
        raise ValueError("eight binary disjoint full-ROI regions required")
    attempts = [json.loads(line) for line in reader.read(root/"attempts.jsonl").splitlines() if line.strip()]
    if len(attempts) != 88 or len({a["id"] for a in attempts}) != 88:
        raise ValueError("exact88 distinct recorded attempts required, no replacement/filtering of failures")
    snapshots = reader.js(root/"RAW_SNAPSHOT_INDEX.json")["snapshots"]
    observed = {s["id"]:s for s in result["states"]}
    if set(observed) != {s["id"] for s in expected} or len(result["states"]) != 11:
        raise ValueError("all11 original states required")
    rows=[]; last_id=last_tick=-1; origin=min(a["host_get_started_unix_ns"] for a in attempts)
    for state in expected:
        obs=observed[state["id"]]; selected=[a for a in attempts if a["state_id"] == state["id"]]
        if ([a["repeat_index"] for a in selected] != list(range(8)) or obs.get("status") != "complete"
                or obs.get("valid_frame_count") != 8 or obs.get("repeat_expected") != 8
                or obs.get("attempt_ids") != [a["id"] for a in selected]
                or any(obs.get("display_readback",{}).get(k,{}).get("matches") is not True
                       or obs["display_readback"][k].get("mismatched_pixels") != 0 for k in ("dmd","slm"))):
            raise ValueError("state repeats/order/display gate differs")
        images=[]; times=[]; identities=[]
        for attempt in selected:
            rp=reader.local(root,attempt["raw"]["path"]); mp=reader.local(root,attempt["metadata"]["path"])
            matching=[s for s in snapshots if reader.local(root,s["archived_folder"]) == mp.parent]
            if rp.name != "CCD_raw.npy" or mp.name != "metadata.json" or rp.parent != mp.parent or len(matching) != 1:
                raise ValueError("raw/metadata snapshot pairing differs")
            snapshot=matching[0]
            if (attempt.get("original_snapshot_folder") != snapshot["original_folder"]
                    or any(any(attempt[key][field] != snapshot["files"][name][field] for field in ("path","sha256"))
                           for key,name in (("raw","CCD_raw.npy"),("metadata","metadata.json")))):
                raise ValueError("indexed original snapshot association differs")
            raw=reader.npy(rp,attempt["raw"]["sha256"]); meta=reader.js(mp,attempt["metadata"]["sha256"]); f=meta["frame"]
            fid,tick,t=f["FrameID"],f["TimeStamp"],f["host_get_started_unix_ns"]
            uid=(plan["camera_expected"]["active"].get("DeviceSerialNumber"),fid,tick)
            if (raw.dtype != np.uint8 or raw.shape != shape or raw.max() >= 240 or f.get("IsIncomplete") is True
                    or fid <= last_id or tick <= last_tick or uid in seen or attempt["status"] != "valid"
                    or f != attempt["frame"] or meta["camera_active"] != plan["camera_expected"]["active"]
                    or meta["display_revision"] != obs["display_revision"] or attempt["display_revision"] != obs["display_revision"]
                    or t < obs["verified_unix_ns"]+1_500_000_000 or attempt.get("input_key") != state["input_key"]
                    or attempt["input_asset"] != plan["inputs"][state["input_key"]]
                    or attempt["fixed_mask_asset"] != plan["masks"]["Z"]):
                raise ValueError("native CCD identity/camera/freshness/revision gate differs")
            images.append(raw); times.append(float(t-origin)); seen.add(uid); last_id,last_tick=fid,tick
            identities.append(dict(attempt_id=attempt["id"],repeat_index=attempt["repeat_index"],raw=attempt["raw"],metadata=attempt["metadata"],frame=f))
        rows.append(dict(**state,stack=np.stack(images),time=float(np.mean(times)),identities=identities))
    return plan,rows,regions


def assess_lowpass(index_path, output, *, expected_index_sha256):
    """All704 old frames, paired sigma study; never alters live/actor processing."""
    import scipy
    out=Path(output); reader=obsa._Reader()
    if out.exists() or out.is_symlink():raise FileExistsError("NEW offline lowpass output required")
    try:
        index=reader.js(index_path,expected_index_sha256,"explicit_historical_E005_index")
        if index.get("schema") != obsa.INDEX_SCHEMA or len(index["captures"]) != 8 or len({r["batch_id"] for r in index["captures"]}) != 8:
            raise ValueError("explicit complete eight-capture E005 index required")
        protected = [*index["captures"], index["bank"], index["predictions"]]
        if any(out.resolve().is_relative_to(Path(r["path"]).resolve()) for r in protected):
            raise ValueError("output cannot modify captured evidence, input bank or frozen predictions")
        out.mkdir();(out/"figures").mkdir();(out/"provenance").mkdir();(out/"derived").mkdir()
        for path in (Path(__file__),Path(obsa.__file__),Path(__file__).with_name("reporting.py"),Path(__file__).with_name("storage.py"),Path(__file__).with_name("protocol.py")):
            blob=reader.read(path,role="offline_analysis_source");(out/"provenance"/path.name).write_bytes(blob)
        sigmas=[dict(sigma=s,states=[],batches=[]) for s in SIGMAS]; feature_files={}; seen=set(); sample={}; sample_identity=None; mixing=[]
        for bi,ref in enumerate(index["captures"]):
            plan,states,regions=_load_capture(reader,ref,index,seen)
            spec=plan["batch_spec"]
            if (spec["mask_key"],spec["radius"],spec["axis_family"]) != GROUP_ORDER[bi]:raise ValueError("predeclared eight-batch design order differs")
            labels=regions.argmax(0).ravel(); saved={}; working={}; rawtotals={}
            if bi == 0:
                area=regions.sum((1,2))
                for s in SIGMAS:
                    effective=spatial_lowpass(regions.astype(float),s)
                    matrix=np.einsum("ihw,jhw->ij",regions,effective)/area[None,:]
                    mixing.append(dict(sigma=s,uniform_within_source_region_software_mixing_matrix=matrix.tolist(),
                        max_column_mass_error=float(np.max(np.abs(matrix.sum(0)-1))),
                        effective_weight_relative_L1_change=(np.abs(effective-regions).sum((1,2))/area).tolist(),
                        meaning="software operator applied to uniform regional basis, NOT measured optical crosstalk or efficiency"))
            for si,s in enumerate(SIGMAS):
                arrays=[]; state_result=[]; work={}
                for state in states:
                    pixels=spatial_lowpass(state["stack"],s)
                    features=_features(pixels,labels)
                    # Translation-invariant sample variance, centered on one
                    # actual repeat: identical float frames then have exact
                    # zero noise instead of summation-roundoff pseudo-noise.
                    cov=np.cov(features-features[:1],rowvar=False,ddof=1)
                    pixel_variance=np.var(pixels-pixels[:1],axis=0,ddof=1)
                    totals=pixels.sum((1,2));rawtotal=state["stack"].astype(float).sum((1,2))
                    values=dict(batch_id=ref["batch_id"],state_id=state["id"],kind=state["kind"],n_frames=8,
                        pixel_temporal_std_RMS_ADU=float(np.sqrt(np.mean(pixel_variance))),
                        native8_mean_ADU=features.mean(0).tolist(),native8_single_frame_covariance_ADU2=cov.tolist(),
                        native8_single_frame_sqrttrace_ADU=float(np.sqrt(np.trace(cov))),
                        raw_mean_total_ADU=float(rawtotal.mean()),filtered_mean_total_ADU=float(totals.mean()),
                        max_abs_per_frame_total_change_ADU=float(np.max(np.abs(totals-rawtotal))),
                        effective_host_time_relative_ns=state["time"])
                    if si == 0:rawtotals[state["id"]]=features
                    delta=features-rawtotals[state["id"]]
                    values["feature_change_vs_raw_RMS_ADU"]=float(np.sqrt(np.mean(delta*delta)))
                    state_result.append(values);arrays.append(features)
                    work[state["id"]]=dict(numeric_eligible=True,mean=features.mean(0),time=state["time"])
                    if bi == 2 and state["id"] == "REF0":
                        sample[s]=pixels[0].copy()
                        if si == 0:sample_identity=dict(batch_id=ref["batch_id"],state_id="REF0",
                            **state["identities"][0],camera_expected=plan["camera_expected"],manual_geometry_expected=plan["manual_geometry_expected"])
                steps=[.03,.03] if spec["axis_family"] == "POS" else [.12,.12]
                jac=obsa.local_jacobian(work,spec["axis_family"],steps)
                working[s]=jac;feature_files[ref["batch_id"]+"__sigma_"+str(s).replace(".","p")]=np.concatenate(arrays)
                saved[s]=state_result
            for si,s in enumerate(SIGMAS):
                comparisons=[]
                for current,base in zip(working[s]["pairs"],working[0.]["pairs"]):
                    for sign,field in (("+","plus_response"),("-","minus_response")):
                        if current[field] is None or base[field] is None:raise ValueError("unexpected non-bracketed complete history")
                        a=np.asarray(current[field]);b=np.asarray(base[field]);na=np.linalg.norm(a);nb=np.linalg.norm(b)
                        comparisons.append(dict(axis=current["axis"],sign=sign,raw_response_ADU=b.tolist(),filtered_response_ADU=a.tolist(),
                            raw_norm_ADU=float(nb),filtered_norm_ADU=float(na),norm_retention_vs_raw=_ratio(na,nb),
                            relative_vector_change_vs_raw=_ratio(np.linalg.norm(a-b),nb),cosine_vs_raw=_ratio(a@b,na*nb)))
                for row,base in zip(saved[s],saved[0.]):
                    row["pixel_temporal_std_RMS_ratio_vs_raw"]=_ratio(row["pixel_temporal_std_RMS_ADU"],base["pixel_temporal_std_RMS_ADU"])
                    row["native8_sqrttrace_ratio_vs_raw"]=_ratio(row["native8_single_frame_sqrttrace_ADU"],base["native8_single_frame_sqrttrace_ADU"])
                sigmas[si]["states"].extend(saved[s]);sigmas[si]["batches"].append(dict(batch_id=ref["batch_id"],contrasts=comparisons,jacobian=working[s]))
            print("offline lowpass: sealed "+ref["batch_id"]+" / "+str(bi+1)+" of8",flush=True)
        np.savez_compressed(out/"derived/all704_native8_by_sigma.npz",**feature_files)
        summary=[]
        for row in sigmas:
            changes=[v["relative_vector_change_vs_raw"] for b in row["batches"] for v in b["contrasts"] if v["relative_vector_change_vs_raw"] is not None]
            pixel_ratios=[s["pixel_temporal_std_RMS_ratio_vs_raw"] for s in row["states"]]
            region_ratios=[s["native8_sqrttrace_ratio_vs_raw"] for s in row["states"]]
            summary.append(dict(sigma=row["sigma"],states=88,frames=704,
                median_pixel_temporal_std_ratio=_median(pixel_ratios),pixel_ratio_valid_states=sum(v is not None for v in pixel_ratios),
                median_native8_sqrttrace_ratio=_median(region_ratios),native8_ratio_valid_states=sum(v is not None for v in region_ratios),
                max_abs_per_frame_total_change_ADU=max(s["max_abs_per_frame_total_change_ADU"] for s in row["states"]),
                median_response_vector_change_vs_raw=_median(changes),max_response_vector_change_vs_raw=float(max(changes)) if changes else None,
                response_change_valid_contrasts=len(changes)))
        plt=plotting();fig,axes=plt.subplots(1,4,figsize=(12,4))
        for ax,s in zip(axes,SIGMAS):
            im=ax.imshow(sample[s],cmap="gray",vmin=0,vmax=32);ax.set_title("sigma="+str(s)+" native pixels");ax.set_axis_off()
            ax.contour(regions.argmax(0),levels=np.arange(.5,7.5),colors="#56B4E9",linewidths=.3)
        fig.colorbar(im,ax=list(axes),fraction=.018,pad=.02,label="Same fixed0..32 ADU")
        fig.suptitle("OLD E005 before manual realignment / B02 REF0 repeat0 / spatial only / NOT live or actor filtering",fontsize=10)
        save_figure(fig,out/"figures/01_fixed_real_single_sigma_comparison",180);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11,3.6))
        for ax,key,title in zip(axes,("median_pixel_temporal_std_ratio","median_native8_sqrttrace_ratio","median_response_vector_change_vs_raw"),
            ("Pixel temporal std ratio / raw","Raw8 sqrttrace covariance ratio / raw","Median response-vector change / raw")):
            ax.plot(SIGMAS,[r[key] for r in summary],"o-");ax.set_title(title,fontsize=9);ax.set_xlabel("Gaussian sigma / native pixels")
        fig.suptitle("All88 states / 704 historical single frames | NO sqrtN, NO background subtraction, NO covariance inverse")
        fig.tight_layout();save_figure(fig,out/"figures/02_all_state_noise_and_response_summary",180);plt.close(fig)
        from PIL import Image
        images=[]
        for s in SIGMAS:
            display=np.rint(np.clip(sample[s]/32,0,1)*255).astype(np.uint8);b=io.BytesIO();Image.fromarray(display).save(b,format="PNG")
            images.append("data:image/png;base64,"+base64.b64encode(b.getvalue()).decode())
        viewer="""<!doctype html><meta charset='utf-8'><title>E006 OLD E005 OFFLINE viewer</title><style>body{font:16px sans-serif;max-width:950px;margin:2em auto;background:#eee}img{max-width:90%;image-rendering:auto}button{margin:.3em;padding:.5em}.warn{color:#a00000}</style><h1>重对齐前 E005：离线空间低通</h1><p class='warn'>不是实时CCD；父代理曾见原GUI停采/显示核验故障，未接入滤波。不是重对齐后新证据，不改变actor。默认低通关闭，显示sigma0原图；按钮只切换本地嵌入的旧单帧。</p><p>固定B02/REF0/repeat0，同一原始帧、固定0..32 ADU；σ仅原生像素空间，无时间平均或背景扣除。</p>"""
        viewer+="".join("<button onclick='show("+str(i)+")'>sigma="+str(s)+"</button>"for i,s in enumerate(SIGMAS))
        viewer+="<button onclick='show(0)'>关闭低通（原图）</button><p id='t'>低通关闭 / 原图</p><img id='v' alt='默认低通关闭，显示原图'><script>const imgs="+json.dumps(images)+",sigmas="+json.dumps(SIGMAS)+";function show(i){document.getElementById('v').src=imgs[i];document.getElementById('t').textContent='OLD E005 / sigma='+sigmas[i]+' / fixed0..32 ADU';}show(0);</script>"
        (out/"dashboard.html").write_text(viewer,encoding="utf8")
        result=dict(schema="optical-lab-historical-spatial-lowpass-1",complete=True,analysis_complete=True,index_sha256=expected_index_sha256,
            software_versions={"numpy":np.__version__,"scipy":scipy.__version__},kernel_implementation="scipy.ndimage.gaussian_filter",
            captures=index["captures"],bank_and_predictions_identity_inherited_NOT_reopened={"bank":index["bank"],"predictions":index["predictions"]},
            sigmas=SIGMAS,summary=summary,statistics=sigmas,regional_software_mixing=mixing,display_sample=sample_identity,
            states_per_sigma=88,frames_per_sigma=704,no_frame_or_sigma_selection=True,output_dtype="float64",mode="reflect",truncate=3.,
            temporal_filtering=False,background_subtraction=False,actor_modified=False,live_GUI_modified=False,hardware_accessed=False,
            control_release_approved=False,production_qualified=False,
            history_is_before_user_manual_realignment_20261023=True,not_new_alignment_evidence=True,not_real_time=True,
            current_GUI_status_observed_by_parent_NOT_requeried=dict(pid=2462026,live_camera=False,hardware_fault="显示核验失败：dmd实际显示被覆盖或与请求不同",manual_x=-402,manual_y=-50,historical_x=-405,historical_y=-41),
            qualification={k:"UNKNOWN"for k in("filtering_benefit","model","control","20ms")})
        lines=["# E006：重对齐前E005空间低通离线对照","","用户2026-10-23刚手动重对齐；本批全部来自此前冻结E005，不代表新光路。父代理只读见原GUI PID2462026停采、DMD显示核验故障，当前(-402,-50)，历史(-405,-41)；本程序没有重新读设备或证明现场安全，原GUI也没有接好实时滤波。",
            "","sigma=0/.5/1/2只作用末两轴，float64、SciPy半采样reflect、truncate3。反射只是数字边界假设，不是相机外真实光场。没有背景扣除、时间平均、clip或actor修改。BG仍是DMD/SLM全0光学背景，不是物理遮光dark。",
            "","像素噪声指标为同状态8帧的时间std RMS（ddof1），不是单图空间纹理或均值stderr。8区先对每张滤后图积分再求完整协方差；不把滤后像素当独立、不除sqrtN、不求逆。低通不去除黑电平/DC或全局时间漂移。",
            "","固定BROAD8内大部分积分权重不变；平滑主要把边界变成软权重，可能看着干净却8区噪声几乎不降。附software mixing仅均匀区域基下的数字串扰，不是实测光学效率。每个sigma的全部704×8特征都保存，可复核而不另存大滤波raw。",
            "","| sigma | 像素时间std比 中位数 | 8区sqrttrace比 中位数 | 响应向量相对变化 中位数/最大 |","|---:|---:|---:|---:|"]
        fmt=lambda v:"NA"if v is None else f"{v:.6g}"
        for r in summary:lines.append(f"| {r['sigma']} | {fmt(r['median_pixel_temporal_std_ratio'])} ({r['pixel_ratio_valid_states']}/88) | {fmt(r['median_native8_sqrttrace_ratio'])} ({r['native8_ratio_valid_states']}/88) | {fmt(r['median_response_vector_change_vs_raw'])} / {fmt(r['max_response_vector_change_vs_raw'])} ({r['response_change_valid_contrasts']}/32) |")
        lines += ["","同一旧数据的配对数字实验，不据此选sigma宣称新对齐通过、光学优势或独立SNR；后续新采必须另立身份。原始capture/源文件均未改。","","![固定真实单帧](figures/01_fixed_real_single_sigma_comparison.png)","![全部状态指标](figures/02_all_state_noise_and_response_summary.png)","","[默认低通关闭、显示原图的离线按钮viewer](dashboard.html)"]
        (out/"总结.md").write_text("\n".join(lines)+"\n",encoding="utf8")
        reader.tail();write_json_new(out/"source_sha256.json",list(reader.sources.values()));write_json_new(out/"RESULT.json",result)
        write_json_new(out/"MANIFEST.json",dict(schema=result["schema"],complete=True,file_sha256={str(p.relative_to(out)):sha256(p)for p in out.rglob("*")if p.is_file()},manifest_excludes_itself=True))
        return result
    except Exception as exc:
        if out.exists():write_json_new(out/"FAILED.json",dict(complete=False,error=repr(exc),original_capture_and_actor_not_modified=True))
        raise
