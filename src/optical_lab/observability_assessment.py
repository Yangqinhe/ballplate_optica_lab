"""Sealed E005 local CAL analysis. No SDK, GUI, model inference or fitting.

All means/reference interpolation are offline summaries, never actor inputs.
Eight CCD samples describe one stationary command, not eight image histories.
"""
from pathlib import Path
import hashlib
import io
import json

import numpy as np

from .storage import write_json_new, sha256
from .reporting import plotting, save_figure
from .protocol import observability_states


SCHEMA = "optical-lab-observability-analysis-1"
INDEX_SCHEMA = "optical-lab-observability-analysis-index-1"
BROAD8_SHA256 = "0a76e770d8d9f2761a9edce492b4a846eab98ebf0e6f719271711202954601c5"
ROI = dict(OffsetX=888, OffsetY=416, Width=552, Height=554)
TARGET_TAU = np.array([.01, .01, .02, .02])
PREDICTION_SCOPE = "numeric_arrays_full_initial_preload_final__metadata_each_control_check"
PREDICTION_CONTROL_FILES = {"RESULT.json", "PROSPECTIVE_PROTOCOL.json", "identification_source.py",
    "M0_REFERENCE_PRE_PREDICTION_FREEZE.json", "M1_RADIAL_PRE_PREDICTION_FREEZE.json", "SENSITIVITY_M0_PRE_PREDICTION_FREEZE.json"}


class _Reader:
    def __init__(self):
        self.sources = {}
        self.archives = {}

    def read(self, path, expected=None, role="input"):
        path = Path(path).resolve(strict=True)
        blob = path.read_bytes(); digest = hashlib.sha256(blob).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("SHA changed: " + str(path))
        if str(path) in self.sources and self.sources[str(path)]["sha256"] != digest:
            raise ValueError("file changed during analysis: " + str(path))
        self.sources[str(path)] = dict(path=str(path), sha256=digest, role=role)
        return blob

    def js(self, path, expected=None, role="metadata"):
        return json.loads(self.read(path, expected, role))

    def npy(self, path, expected=None):
        return np.load(io.BytesIO(self.read(path, expected, "array")), allow_pickle=False)

    def archive(self, path, expected):
        root = Path(path).resolve(strict=True)
        manifest = self.js(root / "MANIFEST.json", expected, "externally_bound_seal")
        names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
        if names != set(manifest["file_sha256"]):
            raise ValueError("sealed file inventory changed: " + str(root))
        for name, digest in manifest["file_sha256"].items():
            self.read(self.local(root, name), digest, "sealed_artifact")
        self.archives[str(root)] = (expected, names)
        return root, manifest

    @staticmethod
    def local(root, name):
        p = Path(root) / name
        if Path(name).is_absolute() or ".." in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("archive path escape/link")
        return p

    def tail(self):
        for item in list(self.sources.values()):
            self.read(item["path"], item["sha256"], item["role"])
        for root, (digest, names) in self.archives.items():
            p = Path(root)
            actual = {str(f.relative_to(p)) for f in p.rglob("*") if f.is_file() and f != p / "MANIFEST.json"}
            if actual != names or sha256(p / "MANIFEST.json") != digest:
                raise ValueError("archive changed during analysis")


def _equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def bracketed_response(probe, before, after, probe_time, before_time, after_time):
    """Strict same-clock interpolation; invalid chronology is NA, not extrapolation."""
    if not all(np.isfinite(t) for t in (probe_time, before_time, after_time)) or not before_time < probe_time < after_time:
        return None
    w = (probe_time-before_time)/(after_time-before_time)
    return dict(response=np.asarray(probe, float)-(1-w)*np.asarray(before, float)-w*np.asarray(after, float),
                weights=[float(1-w), float(w)])


def local_jacobian(states, axis_family, steps):
    """Central differences of REF-corrected means; also retain direct raw differences."""
    axes = ("X", "Y") if axis_family == "POS" else ("VX", "VY")
    columns, direct, pairs = [], [], []
    for j, axis in enumerate(axes):
        plus, minus = states.get(axis+"_PLUS"), states.get(axis+"_MINUS")
        refs = (states.get("REF"+str(2*j)), states.get("REF"+str(2*j+1)), states.get("REF"+str(2*j+2)))
        row = dict(axis=axis, status="NA", plus_response=None, minus_response=None,
                   direct_raw_difference=None, central_difference=None, reason="missing/ineligible probe or bracketed REF")
        if all(s is not None and s["numeric_eligible"] for s in (plus, minus, *refs)):
            p = bracketed_response(plus["mean"], refs[0]["mean"], refs[1]["mean"], plus["time"], refs[0]["time"], refs[1]["time"])
            m = bracketed_response(minus["mean"], refs[1]["mean"], refs[2]["mean"], minus["time"], refs[1]["time"], refs[2]["time"])
            if p is not None and m is not None:
                d = (p["response"]-m["response"])/(2*steps[j])
                raw = (plus["mean"]-minus["mean"])/(2*steps[j])
                row.update(status="descriptive", plus_response=p["response"].tolist(), minus_response=m["response"].tolist(),
                    weights_plus=p["weights"], weights_minus=m["weights"], direct_raw_difference=raw.tolist(),
                    central_difference=d.tolist(), reason=None)
                columns.append(d); direct.append(raw)
        pairs.append(row)
    return dict(axes=list(axes), pairs=pairs, jacobian=(np.stack(columns, axis=1).tolist() if len(columns)==2 else None),
                direct_raw_jacobian=(np.stack(direct, axis=1).tolist() if len(direct)==2 else None),
                definition="central difference of probe minus its two adjacent time-interpolated REF means; host clock is NOT verified exposure")


def _bank(reader, ref):
    root, manifest = reader.archive(ref["path"], ref["sha256"])
    bank = reader.js(root / "BANK.json")
    if (bank.get("schema") != "optical-lab-observability-bank-1" or bank.get("hardware_accessed") is not False
            or bank.get("role") != "development_calibration_NOT_blind_validation"):
        raise ValueError("explicit CAL-only observability bank required")
    expected = {f"R{r}_{k}" for r in (0,2) for k in ("REF","X_PLUS","X_MINUS","Y_PLUS","Y_MINUS","VX_PLUS","VX_MINUS","VY_PLUS","VY_MINUS")}
    if set(bank["inputs"]) != expected or set(bank["masks"]) != {"learned","lens","flat"}:
        raise ValueError("complete eighteen inputs / three masks required")
    arrays = {}; raw_arrays = {}
    for key, record in bank["inputs"].items():
        for item in (record, record["raw96"]):
            p=Path(item["path"])
            if p.parent != root or manifest["file_sha256"].get(p.name) != item["sha256"]:
                raise ValueError("bank input identity differs")
        raw=reader.npy(record["raw96"]["path"],record["raw96"]["sha256"])
        dmd=reader.npy(record["path"],record["sha256"])
        packed=np.pad(np.block([[raw[0,0],raw[1,0]],[raw[2,0],raw[3,0]]]).repeat(3,0).repeat(3,1),12)
        if raw.dtype!=np.uint8 or raw.shape!=(4,1,96,96) or not np.isin(raw,[0,255]).all() or dmd.dtype!=np.uint8 or not np.array_equal(dmd,packed):
            raise ValueError("exact binary chronological four-frame DMD packing required")
        if record["id"]!=key or record["history_ages_s"]!=[.18,.12,.06,0.]:raise ValueError("chronological bank input identity differs")
        arrays[key]=dmd;raw_arrays[key]=raw
    for key,raw in raw_arrays.items():
        if key.startswith("R2_"):
            source=raw_arrays["R0_"+key[3:]]
            padded=np.pad(source>0,((0,0),(0,0),(2,2),(2,2)));expected=np.zeros_like(source,dtype=bool)
            for dy in range(-2,3):
                for dx in range(-2,3):
                    if dy*dy+dx*dx<=4:expected|=padded[:,:,2+dy:98+dy,2+dx:98+dx]
            if not np.array_equal(raw,expected.astype(np.uint8)*255):raise ValueError("R2 is not the exact frozen disk dilation of R0")
    masks={}
    for key, record in bank["masks"].items():
        if Path(record["path"]).parent!=root or record["sha256"]!=bank["design"]["masks"][key]["sha256"]:
            raise ValueError("frozen mask identity differs")
        array=reader.npy(record["path"],record["sha256"])
        if array.dtype!=np.uint8 or array.shape!=(600,600) or not np.array_equal(array,array[::4,::4].repeat(4,0).repeat(4,1)):
            raise ValueError("raw canonical600 macro4 mask required")
        masks[key]=array
    for path,digest in bank["source_bindings"].items():reader.read(path,digest,"bank_source_binding")
    return root,bank,arrays,masks


def _patterns(plan, dmd, mask, illuminated):
    """Explicit original canonical600 embed; physical SLM horizontal flip ONCE."""
    g=plan["manual_geometry_expected"]
    if g["full_panel"] is not False or type(g["x"]) is not int or type(g["y"]) is not int or abs(g["x"])>660 or abs(g["y"])>240:
        raise ValueError("unchanged fully-fitting canonical600 window required")
    cd=dmd if illuminated else np.zeros_like(dmd)
    cs=np.zeros((1080,1920),np.uint8)
    if illuminated:cs[240+g["y"]:840+g["y"],660-g["x"]:1260-g["x"]]=mask
    video=np.zeros((600,800),np.uint8);video[:,100:700]=cd
    return dict(dmd=cd,slm=cs),dict(dmd=video,slm=np.ascontiguousarray(np.fliplr(cs)))


def _png(reader,path):
    from PIL import Image
    with Image.open(io.BytesIO(reader.read(path,role="saved_requested_pixels"))) as im:
        return np.asarray(im)


def _capture(reader,ref,bank_ref,bank,pred_ref,inputs,masks,seen):
    root,_=reader.archive(ref["path"],ref["sha256"])
    plan=reader.js(root/"PLAN.json"); result=reader.js(root/"RESULT.json")
    batch=next((b for b in bank["design"]["batch_order"] if b["id"]==ref["batch_id"]),None)
    if batch is None:raise ValueError("undeclared batch")
    family=batch["axis_family"]; expected=observability_states(family)
    names=("REF","X_PLUS","X_MINUS","Y_PLUS","Y_MINUS") if family=="POS" else ("REF","VX_PLUS","VX_MINUS","VY_PLUS","VY_MINUS")
    input_records={k:bank["inputs"][f"R{batch['radius']}_{k}"] for k in names}
    if (plan.get("profile")!="observability_v1" or plan.get("observability_bank")!=bank_ref
            or not _equal(plan.get("batch_spec"),batch) or plan.get("batch_id")!=ref["batch_id"]
            or plan.get("axis_family")!=family or plan.get("mask_key")!=batch["mask_key"]
            or not _equal(plan.get("states"),expected) or type(plan.get("frames_per_state"))is not int
            or plan["frames_per_state"]!=8 or plan.get("total_planned_frames")!=88
            or not _equal(plan.get("inputs"),input_records) or not _equal(plan.get("input"),input_records["REF"])
            or plan.get("prospective_predictions")!={"path":pred_ref["path"],"manifest_sha256":pred_ref["sha256"],"verification_scope":PREDICTION_SCOPE}
            or any(k in plan for k in ("pilot_manifest_sha256","case_asset_manifest","frozen_predictions"))):
        raise ValueError("capture profile/batch/input/prediction identity differs")
    if not _equal(plan["masks"],{k:bank["masks"][batch["mask_key"]] for k in ("Z","A","B")}):
        raise ValueError("one fixed mask for entire batch required")
    required={str(Path(bank_ref["path"])/"MANIFEST.json"):bank_ref["sha256"],
        **{str(Path(bank_ref["path"])/n):h for n,h in reader.js(Path(bank_ref["path"])/"MANIFEST.json")["file_sha256"].items()},
        **bank["source_bindings"],str(Path(pred_ref["path"])/"MANIFEST.json"):pred_ref["sha256"],
        **{str(Path(pred_ref["path"])/n):h for n,h in reader.js(Path(pred_ref["path"])/"MANIFEST.json")["file_sha256"].items()if n in PREDICTION_CONTROL_FILES}}
    if any(plan["source_sha256"].get(p)!=h for p,h in required.items()):
        raise ValueError("capture PLAN omits/changes a bank or prediction source binding")
    for path,digest in plan["source_sha256"].items():reader.read(path,digest,"capture_plan_source_binding")
    if (result.get("profile")!="observability_v1" or result.get("observability_bank")!=bank_ref
            or result.get("batch_id")!=ref["batch_id"] or not _equal(result.get("batch_spec"),batch)
            or result.get("prospective_predictions")!=plan["prospective_predictions"]):
        raise ValueError("capture RESULT batch/bank/prediction identity differs")
    if plan["camera_expected"]["roi"]!=ROI or plan["regions"]["sha256"]!=BROAD8_SHA256:
        raise ValueError("original native ROI/BROAD8 required; no implicit resize or substitution")
    regions=reader.npy(plan["regions"]["path"],BROAD8_SHA256)
    shape=(ROI["Height"],ROI["Width"])
    if regions.shape!=(8,*shape) or not np.isin(regions,[0,1]).all() or not np.all(regions.sum(0)==1):
        raise ValueError("eight fixed binary disjoint full-ROI regions required")
    labels=regions.argmax(0).ravel()
    path=root/"attempts.jsonl"
    attempts=[json.loads(line) for line in reader.read(path).splitlines() if line.strip()] if path.exists() else []
    if len({r["id"] for r in attempts})!=len(attempts) or any(r["state_id"]not in {s["id"]for s in expected}for r in attempts):
        raise ValueError("duplicate attempt or undeclared state")
    snapshot_path=root/"RAW_SNAPSHOT_INDEX.json"
    snapshots=reader.js(snapshot_path).get("snapshots",[])if snapshot_path.exists()else[]
    snapshot_by_folder={}
    originals=set()
    for snapshot in snapshots:
        folder=reader.local(root,snapshot["archived_folder"])
        original=snapshot["original_folder"]
        if folder in snapshot_by_folder or original in originals or not Path(original).is_absolute():
            raise ValueError("duplicate/ambiguous snapshot index identity")
        for filename,item in snapshot["files"].items():
            p=reader.local(root,item["path"])
            if (p.parent!=folder or p.name!=filename or item.get("original_path")!=str(Path(original)/filename)
                    or reader.sources.get(str(p.resolve()),{}).get("sha256")!=item["sha256"]):
                raise ValueError("snapshot index file/folder/SHA association differs")
        snapshot_by_folder[folder]=snapshot;originals.add(original)
    observed={s["id"]:s for s in result.get("states",[])}
    if len(observed)!=len(result.get("states",[])) or set(observed)-{s["id"]for s in expected}:raise ValueError("duplicate/unknown saved state")
    host=[a.get("host_get_started_unix_ns")for a in attempts if type(a.get("host_get_started_unix_ns"))is int]
    origin=min(host)if host else None
    states=[];features={};working={};last_id=last_tick=-1
    for spec in expected:
        sid=spec["id"];obs=observed.get(sid,{});rows=[a for a in attempts if a["state_id"]==sid]
        reject=[];values=[];times=[]
        cd,physical=_patterns(plan,inputs[f"R{batch['radius']}_{spec['input_key']}"],masks[batch["mask_key"]],spec["kind"]!="BG")
        display=obs.get("display_readback",{})
        display_ok=all(display.get(k,{}).get("matches")is True and display[k].get("mismatched_pixels")==0 for k in ("dmd","slm"))
        expected_panel={k:dict(shape_yx=list(v.shape),dtype="uint8_grayscale_equal_RGB",pixel_sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in physical.items()}
        display_ok=display_ok and _equal(obs.get("expected_panel"),expected_panel)
        for row in rows:
            if (row.get("kind")!=spec["kind"] or row.get("time_block")is not None):
                raise ValueError("attempt kind differs from declared INPUT/BG sequence")
            if row.get("status")!="valid" or not row.get("raw")or not row.get("metadata"):
                reject.append(dict(id=row["id"],reason=row.get("error")or"not a valid saved attempt"));continue
            rawpath=reader.local(root,row["raw"]["path"]);metapath=reader.local(root,row["metadata"]["path"])
            folder=metapath.parent;snapshot=snapshot_by_folder.get(folder)
            required_files={"CCD_raw.npy","CCD_raw.png","metadata.json","dmd_canonical_requested.png","slm_canonical_requested.png"}
            if (rawpath.name!="CCD_raw.npy" or metapath.name!="metadata.json" or rawpath.parent!=folder
                    or snapshot is None or row.get("original_snapshot_folder")!=snapshot["original_folder"]
                    or not required_files.issubset(snapshot["files"])
                    or any(any(row[k][field]!=snapshot["files"][name][field]for field in("path","sha256"))
                           for k,name in(("raw","CCD_raw.npy"),("metadata","metadata.json")))):
                raise ValueError("raw/metadata must be paired in one indexed original snapshot folder")
            raw=reader.npy(rawpath,row["raw"]["sha256"])
            meta=reader.js(metapath,row["metadata"]["sha256"])
            if raw.dtype!=np.uint8 or raw.shape!=shape:raise ValueError("native uint8 raw layout changed")
            frame=meta.get("frame",{});fid=frame.get("FrameID");tick=frame.get("TimeStamp");t=frame.get("host_get_started_unix_ns")
            uid=(plan["camera_expected"]["active"].get("DeviceSerialNumber"),fid,tick)
            wait=plan.get("minimum_postcommit_frame_start_seconds")
            good=(type(fid)is int and type(tick)is int and type(t)is int and fid>last_id and tick>last_tick
                and uid not in seen and frame.get("IsIncomplete")is not True and int(raw.max())<240
                and frame.get("Width")==shape[1] and frame.get("Height")==shape[0]
                and frame.get("Stride")==shape[1] and frame.get("BufferSize")==shape[0]*shape[1] and frame.get("pixel_format")=="Mono8"
                and type(obs.get("verified_unix_ns"))is int and type(wait)in(int,float)and wait>0
                and t>=obs["verified_unix_ns"]+int(wait*1e9)
                and meta.get("camera_active")==plan["camera_expected"]["active"] and row.get("camera_actual")==meta.get("camera_active")
                and _equal(row.get("frame"),frame)and row.get("host_get_started_unix_ns")==t
                and meta.get("display_revision")==obs.get("display_revision")==row.get("display_revision")
                and row.get("input_key")==spec["input_key"] and _equal(row.get("input_asset"),input_records[spec["input_key"]])
                and row.get("mask_key")==batch["mask_key"] and row.get("fixed_mask_asset")==plan["masks"]["Z"])
            if not good:
                reject.append(dict(id=row["id"],reason="independent layout/camera/identity/freshness/revision/headroom gate"));continue
            for kind in ("dmd","slm"):
                filename=kind+"_canonical_requested.png"
                if (reader.local(root,snapshot["files"][filename]["path"])!=folder/filename
                        or not np.array_equal(_png(reader,folder/filename),cd[kind])):
                    raise ValueError("saved canonical requested pixels differ from exact input/mask")
            seen.add(uid);last_id,last_tick=fid,tick
            values.append(np.bincount(labels,weights=raw.ravel(),minlength=8));times.append(float(t-origin))
        array=np.asarray(values,dtype=np.float64).reshape(-1,8);features[sid]=array
        complete=(len(values)==8 and len(rows)==8 and not reject and [a["repeat_index"]for a in rows]==list(range(8))
            and obs.get("status")=="complete"and obs.get("valid_frame_count")==8 and obs.get("repeat_expected")==8
            and obs.get("attempt_ids")==[r["id"]for r in rows]and display_ok)
        mean=array.mean(0)if len(array)else None;cov=np.cov(array,rowvar=False,ddof=1)if len(array)>1 else None
        time=float(np.mean(times))if times else None
        working[sid]=dict(numeric_eligible=bool(complete),mean=mean,time=time)
        states.append(dict(**spec,numeric_eligible=bool(complete),valid_frames=len(array),rejected=reject,
            display_expected_pixels_verified=bool(display_ok),native8_mean_ADU=None if mean is None else mean.tolist(),
            native8_single_frame_covariance_ADU2=None if cov is None else cov.tolist(),
            single_frame_repeat_scale_ADU=None if cov is None else float(np.sqrt(np.trace(cov))),
            effective_host_time_relative_ns=time,verified_exposure_time=False,reported_status=obs.get("status","not_started")))
    steps=bank["design"]["step_sizes"][:2]if family=="POS"else bank["design"]["step_sizes"][2:]
    jac=local_jacobian(working,family,steps)
    refs=[s for s in states if s["id"].startswith("REF")]
    reference_drift=[]
    for s in refs:
        first=refs[0]
        reference_drift.append(dict(id=s["id"],delta_from_REF0_ADU=(None if not(s["numeric_eligible"]and first["numeric_eligible"])else(np.array(s["native8_mean_ADU"])-first["native8_mean_ADU"]).tolist())))
    final_ok=(result.get("complete")is True and all(s["numeric_eligible"]for s in states)
        and all(result.get(k)is True for k in("actual_outputs_zero","watchdog_verified","camera_left_running","frozen_inputs_unchanged"))
        and not any(result.get(k)for k in("error","cleanup_error","archive_error","integrity_error","stopped_by_bound","resource_cleanup_errors")))
    return dict(batch_id=ref["batch_id"],status="COMPLETE"if final_ok else"INCOMPLETE",
        capture=ref,capture_declared_complete=result.get("complete")is True,capture_error=result.get("error"),
        planned_frames=88,valid_frames=sum(s["valid_frames"]for s in states),states=states,reference_drift=reference_drift,
        time_origin_unix_ns=origin,jacobian=jac,axis_family=family,mask_key=batch["mask_key"],radius=batch["radius"],
        camera_expected=plan["camera_expected"],manual_geometry_expected=plan["manual_geometry_expected"],
        capture_actual_outputs_zero=result.get("actual_outputs_zero"),watchdog_verified=result.get("watchdog_verified")),features


def assess(index_path,output,expected_index_sha256):
    """Index: bank{path,sha256},predictions{path,sha256},captures[{batch_id,path,sha256}]."""
    reader=_Reader();out=Path(output)
    if out.exists()or out.is_symlink():raise FileExistsError("NEW observability report required")
    try:
        index=reader.js(index_path,expected_index_sha256,"postcapture_identity_index_NOT_prospective_design")
        if index.get("schema")!=INDEX_SCHEMA:raise ValueError("explicit observability analysis index required")
        root,bank,inputs,masks=_bank(reader,index["bank"])
        pred,predmanifest=reader.archive(index["predictions"]["path"],index["predictions"]["sha256"])
        predresult=reader.js(pred/"RESULT.json")
        if (predresult.get("complete")is not True or predresult.get("hardware_accessed")is not False
                or predresult.get("bank_sha256")!=index["bank"]["sha256"] or predresult.get("nominal_prediction_count")!=108
                or predresult.get("all_nominal_means_below_capture_limit240")is not True
                or not PREDICTION_CONTROL_FILES.issubset(predmanifest["file_sha256"])):
            raise ValueError("complete frozen predictions bound to exact bank required")
        ids=[s["id"]for s in bank["design"]["batch_order"]];refs=index["captures"]
        if len({r["batch_id"]for r in refs})!=len(refs)or any(r["batch_id"]not in ids for r in refs):raise ValueError("duplicate/undeclared capture batch")
        for tree in [root,pred,*[Path(r["path"]).resolve()for r in refs]]:
            if out.resolve().is_relative_to(tree):raise ValueError("output must not modify an input artifact tree")
        source_blobs={}
        for path in (Path(__file__),Path(__file__).with_name("protocol.py"),Path(__file__).with_name("reporting.py"),Path(__file__).with_name("storage.py")):
            source_blobs[path.name]=reader.read(path,role="analysis_source")
        out.mkdir();(out/"figures").mkdir();(out/"derived").mkdir();(out/"provenance").mkdir()
        for name,blob in source_blobs.items():(out/"provenance"/name).write_bytes(blob)
        byid={r["batch_id"]:r for r in refs};batches=[];features={};seen=set()
        for spec in bank["design"]["batch_order"]:
            if spec["id"]not in byid:
                empty_states=[dict(**s,numeric_eligible=False,valid_frames=0,reported_status="not_started",rejected=[],
                    native8_mean_ADU=None,native8_single_frame_covariance_ADU2=None,single_frame_repeat_scale_ADU=None,
                    effective_host_time_relative_ns=None)for s in observability_states(spec["axis_family"])]
                batches.append(dict(batch_id=spec["id"],status="NOT_EXECUTED",planned_frames=88,valid_frames=0,states=empty_states,jacobian=None,
                    mask_key=spec["mask_key"],radius=spec["radius"],axis_family=spec["axis_family"]));continue
            batch,arrays=_capture(reader,byid[spec["id"]],index["bank"],bank,index["predictions"],inputs,masks,seen)
            batches.append(batch)
            for sid,value in arrays.items():features[spec["id"]+"__"+sid]=value
        conditions=[]
        for key,radius in dict.fromkeys((b["mask_key"],b["radius"])for b in batches):
            pair=[b for b in batches if(b["mask_key"],b["radius"])==(key,radius)]
            pos=next(b for b in pair if b["axis_family"]=="POS");vel=next(b for b in pair if b["axis_family"]=="VEL")
            matrix=None;singular=None
            contracts_equal=(pos.get("camera_expected")is not None and pos.get("camera_expected")==vel.get("camera_expected")
                and pos.get("manual_geometry_expected")==vel.get("manual_geometry_expected"))
            if (contracts_equal and pos["status"]==vel["status"]=="COMPLETE" and pos["jacobian"]and vel["jacobian"]
                    and pos["jacobian"]["jacobian"]is not None and vel["jacobian"]["jacobian"]is not None):
                matrix=np.column_stack((pos["jacobian"]["jacobian"],vel["jacobian"]["jacobian"]))
                singular=np.linalg.svd(matrix*TARGET_TAU[None,:],compute_uv=False).tolist()
            conditions.append(dict(mask_key=key,radius=radius,batches=[pos["batch_id"],vel["batch_id"]],
                jacobian_8x4=None if matrix is None else matrix.tolist(),scaled_target_tau=TARGET_TAU.tolist(),
                scaled_jacobian_singular_values_ADU=singular,
                POS_and_VEL_are_sequential_NOT_simultaneous=True,
                batch_time_origins_unix_ns=[pos.get("time_origin_unix_ns"),vel.get("time_origin_unix_ns")],
                complete_pair=pos["status"]==vel["status"]=="COMPLETE",camera_geometry_contracts_equal=bool(contracts_equal),qualification="UNKNOWN"))
        plt=plotting();fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        for i,b in enumerate(batches):
            for s in b["states"]:
                if not s["numeric_eligible"]:continue
                x=i+(int(s["id"][3:])*.025 if s["id"].startswith("REF")else 0)
                axes[0].scatter(x,np.sum(s["native8_mean_ADU"]),s=13,color="#222222"if s["id"].startswith("REF")else"#0072B2")
                axes[1].scatter(x,s["single_frame_repeat_scale_ADU"],s=13,color="#D55E00")
        axes[0].set_ylabel("Native ROI sum / raw ADU sum");axes[1].set_ylabel("Single-frame sqrt(trace(sample covariance))\nNO sqrtN / NO covariance inverse")
        axes[1].set_xticks(range(len(batches)),[b["batch_id"]+"\n"+b["status"]for b in batches],rotation=40,ha="right",fontsize=7)
        fig.suptitle("All planned local CAL batches / BG is optical zero, NOT physical dark / no qualification")
        fig.tight_layout();save_figure(fig,out/"figures/01_states_and_single_frame_variation",180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for c in conditions:
            name=c["mask_key"]+" R"+str(c["radius"])
            if c["scaled_jacobian_singular_values_ADU"]is not None:
                axes[0].plot(range(1,5),c["scaled_jacobian_singular_values_ADU"],"o-",label=name)
                axes[1].plot(np.linalg.norm(np.asarray(c["jacobian_8x4"])*TARGET_TAU[None,:],axis=0),"o-",label=name)
        axes[0].set_ylabel("SVD(J diag(target tau)) / ADU sum");axes[0].set_xlabel("singular-value index; NOT noise-whitened")
        axes[1].set_xticks(range(4),["x .01m","y .01m","vx .02m/s","vy .02m/s"]);axes[1].set_ylabel("Scaled column norm / ADU sum")
        if any(c["scaled_jacobian_singular_values_ADU"]is not None for c in conditions):axes[0].legend(frameon=False)
        fig.suptitle("Local finite differences only / partial pairs NA / sequential masks and encodings, NOT full factorial")
        fig.tight_layout();save_figure(fig,out/"figures/02_local_J_and_SVD",180);plt.close(fig)
        fig,axes=plt.subplots(len(batches),2,figsize=(12,2.0*len(batches)),squeeze=False)
        for row,b in zip(axes,batches):
            row[0].set_title(b["batch_id"]+" / "+b["status"],fontsize=8)
            if b["jacobian"]is None:
                row[0].text(.5,.5,"NA: NOT EXECUTED",ha="center",transform=row[0].transAxes)
            else:
                for pair in b["jacobian"]["pairs"]:
                    for sign,field,style in(("+","plus_response","-"),("-","minus_response","--")):
                        if pair[field]is not None:row[0].plot(range(8),pair[field],style,marker=".",label=pair["axis"]+sign)
                if any(p["status"]=="descriptive"for p in b["jacobian"]["pairs"]):row[0].legend(frameon=False,fontsize=6,ncol=2)
                drift=b["reference_drift"]
                row[1].plot(range(5),[np.linalg.norm(d["delta_from_REF0_ADU"])if d["delta_from_REF0_ADU"]is not None else np.nan for d in drift],"o-",color="#222222",label="mean REF - REF0 norm")
                refs=[s for s in b["states"]if s["id"].startswith("REF")]
                row[1].plot(range(5),[s["single_frame_repeat_scale_ADU"]if s["numeric_eligible"]else np.nan for s in refs],"s--",color="#D55E00",label="single-frame repeat scale")
                row[1].legend(frameon=False,fontsize=6)
            row[0].axhline(0,color=".7",lw=.5);row[0].set_ylabel("probe - adjacent REF / ADU sum")
            row[1].set_ylabel("8-norm / ADU sum");row[1].set_xticks(range(5),[f"REF{i}"for i in range(5)],fontsize=7)
        axes[-1,0].set_xlabel("fixed native8 region index")
        fig.suptitle("All +/- input responses and all reference changes | offline means, NOT independent repetitions\nSingle-frame scale is NOT mean uncertainty or confidence; partial evidence remains labelled",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.96));save_figure(fig,out/"figures/03_all_signed_responses_and_REF_drift",180);plt.close(fig)
        np.savez_compressed(out/"derived/native8_single_frames.npz",**features)
        result=dict(schema=SCHEMA,analysis_complete=True,index_sha256=expected_index_sha256,bank=index["bank"],predictions=index["predictions"],
            predictions_used_for_model_errors=False,model_error_status="NOT_COMPUTED: prediction identity retained only",
            planned_batches=len(ids),planned_frames=88*len(ids),batches=batches,conditions=conditions,
            valid_frames=sum(b["valid_frames"]for b in batches),all_planned_captures_complete=all(b["status"]=="COMPLETE"for b in batches),
            single_frame_covariance_definition="within-state ddof1 sample covariance; serial dependence possible, no sqrtN reduction",
            local_reference_interpolation="same host_get_started clock, NOT verified exposure timing; offline mean-only diagnostic",
            no_covariance_inverse=True,no_head_or_physics_fitting=True,no_model_selection=True,no_hardware_access=True,
            all_inputs_CAL_development_NOT_holdout=True,not_full_factorial=True,production_qualified=False,control_release_approved=False,
            qualification={k:"UNKNOWN"for k in("acquisition","local_observability","model","control","20ms")})
        lines=["# E005 局部输入可观测性：离线归档复算","","全部固定批次保留；不完整/缺失为NA，绝不判PASS。九个prescribed输入都是CAL开发，不是自由动力学或独立时间验证。",
            "","网络仍四幅历史、单当前CCD、固定BROAD8；这里的8张是静态重复样本。单帧8区协方差为ddof1描述，不除sqrtN、不求逆；相邻REF时间插值仅离线漂移诊断。",
            "","J为(plus减夹REF − minus减夹REF)/(2h)，并保存未扣REF的直接差分。SVD仅J乘任务尺度[.01m,.01m,.02m/s,.02m/s]，不是噪声白化/分辨率通过/全非线性可观测性证明。",
            "","learned/lens/flat与编码、时段非完整factorial；旧全灰级mask尚未由本轮实测认定。冻结M0/M1预测仅核身份，未计算模型误差或选模。",
            "","| Batch | 状态 | 完整有效/计划帧 |","|---|---|---:|"]
        for b in batches:lines.append(f"| {b['batch_id']} | {b['status']} | {b['valid_frames']}/88 |")
        lines += ["","POS与VEL两半为分时批次，不是同步四维扰动；同mask/radius合并仅在两批完整且camera/geometry合同一致时进行，不消除批间非线性漂移。", "", "![完整状态与单帧变化](figures/01_states_and_single_frame_variation.png)","![局部J和任务尺度SVD](figures/02_local_J_and_SVD.png)","![全部正负响应与REF变化](figures/03_all_signed_responses_and_REF_drift.png)"]
        (out/"总结.md").write_text("\n".join(lines)+"\n",encoding="utf8")
        reader.tail();write_json_new(out/"source_sha256.json",list(reader.sources.values()));write_json_new(out/"RESULT.json",result)
        write_json_new(out/"MANIFEST.json",dict(schema=SCHEMA,analysis_complete=True,file_sha256={str(p.relative_to(out)):sha256(p)for p in out.rglob("*")if p.is_file()},manifest_excludes_itself=True))
        return result
    except Exception as exc:
        if out.exists():write_json_new(out/"FAILED.json",dict(schema=SCHEMA,analysis_complete=False,error=repr(exc),input_archives_unchanged_by_analysis=True))
        raise
