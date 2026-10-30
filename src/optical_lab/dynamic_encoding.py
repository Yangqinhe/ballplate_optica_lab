"""Same R2 pixel algorithm, explicit quarter-pixel DYNAMIC quality policy.

Import is stdlib-only. No GUI, SDK, optical hardware, state feedback or fit.
The original static encoder and its0.15 limit/qualifications remain unchanged.
Both simulator child and query-verifying parent must explicitly bind this source.
"""
import hashlib
import io
from pathlib import Path
import re

POLICY = 'optical-lab-dynamic-R2-centroid-quarter-RAW-pixel-1'
OLD_STATIC_CENTROID_LIMIT_RAW_PX = .15
DYNAMIC_HARD_CENTROID_LIMIT_RAW_PX = .25
QUERY_TIMEOUT_SECONDS = 30
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def encode_history_dynamic(raw, radius=2):
    """Exact old disk-R2 bytes; old0.15 warning, explicit hard0.25 cutoff.

    Geometry stays four chronological uint8 binary96² frames. No recentering,
    interpolation, threshold change, area change or electronic image network.
    Empty/multi-component ball, edge margin<2, or centroid delta>0.25 reject.
    Returns (encoded, four per-frame geometry/policy metadata dictionaries).
    """
    import numpy as np
    from .study import _component_count
    if type(radius) is not int or radius != 2:
        raise ValueError('dynamic policy is fixed disk R2 only')
    if (not isinstance(raw, np.ndarray) or raw.shape != (4,1,96,96)
            or raw.dtype != np.uint8 or not np.isin(raw,(0,255)).all()):
        raise ValueError('four exact binary uint8 RAW96 frames required')
    encoded=np.zeros_like(raw);rows=[]
    offsets=[(dy,dx)for dy in range(-2,3)for dx in range(-2,3)if dy*dy+dx*dx<=4]
    for i,image in enumerate(raw[:,0]):
        mask=image>0
        if not mask.any()or _component_count(mask)!=1:
            raise ValueError('each source frame must have one nonempty8-connected ball')
        grown=np.zeros_like(mask);padded=np.pad(mask,2)
        for dy,dx in offsets:
            grown|=padded[2+dy:2+dy+96,2+dx:2+dx+96]
        if _component_count(grown)!=1:
            raise ValueError('encoded ball must remain one component')
        old_yx,new_yx=np.argwhere(mask),np.argwhere(grown)
        old_center,new_center=old_yx.mean(0),new_yx.mean(0);delta=new_center-old_center
        norm=float(np.linalg.norm(delta));lo,hi=new_yx.min(0),new_yx.max(0)
        margin=int(min(*lo,*(95-hi)))
        if margin<2:
            raise ValueError('dynamic R2 would clip/approach edge: margin below2 RAW pixels')
        if norm>DYNAMIC_HARD_CENTROID_LIMIT_RAW_PX:
            raise ValueError('dynamic R2 centroid shift exceeds hard0.25 RAW pixels')
        encoded[i,0]=grown.astype(np.uint8)*255
        warning=norm>OLD_STATIC_CENTROID_LIMIT_RAW_PX
        rows.append(dict(frame_index=i,source_on=int(mask.sum()),encoded_on=int(grown.sum()),
            area_ratio=float(grown.sum()/mask.sum()),source_centroid_xy=old_center[::-1].tolist(),
            encoded_centroid_xy=new_center[::-1].tolist(),centroid_delta_xy=delta[::-1].tolist(),
            centroid_delta_L2_raw_px=norm,
            encoded_bbox_xyxy=[int(lo[1]),int(lo[0]),int(hi[1]+1),int(hi[0]+1)],
            minimum_edge_margin_raw_px=margin,connected_components_8=1,
            encoding_quality_policy=POLICY,
            old_static_centroid_L2_limit_raw_px=OLD_STATIC_CENTROID_LIMIT_RAW_PX,
            dynamic_hard_centroid_L2_limit_raw_px=DYNAMIC_HARD_CENTROID_LIMIT_RAW_PX,
            old_limit_exceeded=warning,
            warning='old_static_centroid_limit_exceeded'if warning else None,
            pixel_algorithm_changed=False,old_static_qualification_inherited=False))
    return encoded,rows


def _bound_array(path,expected,child_dir):
    import numpy as np
    p=Path(path)
    if (not p.is_absolute()or not isinstance(expected,str)or not _SHA.fullmatch(expected)
            or p.parent!=child_dir or p.is_symlink()):
        raise ValueError('SHA-bound asset inside dedicated child output required')
    blob=p.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected:
        raise ValueError('frozen simulator asset changed: '+str(p))
    # Parse the exact bytes just hashed, not a second independently read file.
    return np.load(io.BytesIO(blob),allow_pickle=False)


def query_arrays_dynamic(query,index,plan,child_dir,now_ns):
    """E023226..254 query gates, replacing ONLY encoder quality-policy call.

    Same index, feature contract,20ms sim-time,30s host freshness, four explicit
    asset paths/SHA, child-dir, binary layout, exact R2/pack, occupancy and RGB
    gates. No GUI-dependent legacy pin helper is imported. Returns four arrays,
    matching the original parent API; geometry warnings are available by calling
    encode_history_dynamic(source_raw96) and should be archived by the caller.
    """
    import numpy as np
    from .processed8_bank import pack_dmd
    if (query.get('kind')!='optical_query'or type(query.get('index'))is not int or query['index']!=index
            or query.get('feature_contract_sha256')!=plan['feature_contract_sha256']
            or query.get('simulation_time_s')!=index*.02 or type(query.get('created_unix_ns'))is not int
            or plan.get('query_timeout_seconds')!=QUERY_TIMEOUT_SECONDS
            or not 0<=now_ns-query['created_unix_ns']<QUERY_TIMEOUT_SECONDS*1_000_000_000):
        raise ValueError('fresh bound sequential simulator query required')
    child_dir=Path(child_dir);arrays={}
    for name in('source_raw96','raw96','dmd','mujoco_rgb'):
        arrays[name]=_bound_array(query[name],query[name+'_sha256'],child_dir)
    original,encoded,dmd=arrays['source_raw96'],arrays['raw96'],arrays['dmd']
    if any(a.dtype!=np.uint8 or a.shape!=(4,1,96,96)or not np.isin(a,(0,255)).all()for a in(original,encoded)):
        raise ValueError('four chronological binary RAW96 images required')
    expected,_=encode_history_dynamic(original,2)
    if (not np.array_equal(expected,encoded)or dmd.dtype!=np.uint8 or dmd.shape!=(600,600)
            or not np.array_equal(dmd,pack_dmd(encoded))
            or np.count_nonzero(dmd)>plan['allowed_DMD_nonzero_max']):
        raise ValueError('exact bounded dynamic-R2 dilation/packing required; no runtimeR0 substitution')
    rgb=arrays['mujoco_rgb']
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('MuJoCo RGB evidence required')
    return arrays
