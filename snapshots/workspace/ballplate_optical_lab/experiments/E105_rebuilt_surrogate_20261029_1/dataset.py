"""Streaming SHA audit and small derived caches; original RAW is never copied."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from workflow import ROOT, LAB, read, write, ref, require


def checked(reference, array=False):
    import hashlib
    blob=Path(reference['path']).read_bytes()
    require(hashlib.sha256(blob).hexdigest()==reference['sha256'], 'artifact SHA mismatch: '+reference['path'])
    return np.load(io.BytesIO(blob),allow_pickle=False) if array else json.loads(blob)


def integrate(image, layout):
    a=np.asarray(image,np.float64)
    ii=np.pad(a.cumsum(-2).cumsum(-1),((1,0),(1,0)))
    x,y,u,v=np.array([r['bounds'] for r in layout['regions']],int).T
    return ii[v,u]-ii[y,u]-ii[v,x]+ii[y,x]


def build(batches, output):
    output.mkdir(); baseline=read(ROOT/'BASELINE_01/BASELINE.json')
    rows=[];means=[];single=[];offsets=[0];layout=None;sources=[];used_paths=set();peak=0;clipped=0
    for batch in batches:
        folder=ROOT/batch;receipt=read(folder/'RESULT.json');request=read(folder/'REQUEST.json')
        role=checked(request['catalog']).get('new_platform_role','PILOT')
        require(receipt['completed'] and receipt['elapsed_seconds']<=300, 'complete bounded batch required')
        source=receipt['receipt']['result'];result=checked(source);sources += [ref(folder/'RESULT.json'),source]
        contract=checked(result['contract'])
        require(result['shared_main']['camera_actual']==baseline['expected']['camera_actual'], 'new camera contract differs')
        if layout is None:layout=result['region_layout']
        require(result['region_layout']==layout, 'readout layout changed')
        require(result['all_visits_completed'] and result['final_actual_zero_verified'], 'unsealed GUI completion')
        labels=result['input_labels']
        for visit in result['visits']:
            archive=checked(visit['archive'])
            require(archive['completed'] and archive['error'] is None and archive['source_recheck_error'] is None, 'RAW burst failed')
            n=visit['frames'];require(archive['valid_frames']==archive['captured_frames']==n, 'burst length differs')
            acc=np.zeros((layout['roi']['Height'],layout['roi']['Width']),np.float64)
            ff=[]; frame_ids=[]; stamps=[]; frame_refs=[]
            for frame in archive['frames']:
                require(frame['valid'] and frame['raw']['path'] not in used_paths, 'invalid or reused RAW file')
                raw=checked(frame['raw'],True);meta=checked(frame['metadata'])
                require(raw.shape==acc.shape and raw.dtype==np.uint8, 'native RAW shape/type differs')
                require(meta['valid'] and meta['new_research_frame_valid'] and meta['rejection'] is None, 'invalid original frame')
                require(meta['active_readback_before']==meta['active_readback_after']==baseline['expected']['camera_actual'], 'per-frame camera changed')
                frozen=lambda lease:{k:v for k,v in lease.items() if k!='updated_monotonic_ns'}
                require(frozen(meta['lease_before'])==frozen(meta['lease_after'])==frozen(archive['initial_lease']),
                        'pattern identity/revision changed within burst')
                require(meta['raw']==frame['raw'], 'raw/metadata association differs')
                original=meta['original_metadata']
                require(meta['lease_before']['armed'] and not meta['lease_before']['fault'], 'unarmed/faulted lease')
                require(original['host_get_started_unix_ns']>=meta['lease_before']['min_frame_start_ns'], 'pre-settle frame')
                require(meta['readback_before_ns']<=original['host_get_started_unix_ns']<=original['host_received_unix_ns']
                        <=meta['readback_after_ns'], 'SDK chronology differs')
                frame_ids.append(original['FrameID']);stamps.append(original['host_received_unix_ns'])
                used_paths.add(frame['raw']['path']);frame_refs.append(frame)
                peak=max(peak,int(raw.max()));clipped+=int((raw==255).sum())
                acc+=raw;ff.append(integrate(raw,layout))
            require(all(b>a for a,b in zip(stamps,stamps[1:])), 'capture host times not increasing')
            mean=(acc/n).astype(np.float32);z=integrate(mean,layout)
            require(np.max(np.abs(z-np.asarray(visit['raw'])))<1e-4, 'recomputed RAW sum differs from GUI')
            require(np.array_equal(mean,checked(visit['effective_file'],True)), 'effective frame differs from exact mean')
            rows.append(dict(batch=batch,role=role,input_id=visit['input_id'],label=labels[visit['input_id']],
                mask=request['mask'],n=n,archive=visit['archive'],frames=frame_refs,
                mask_name=request.get('mask_name',Path(request['mask']['path']).stem),
                coefficients=request.get('coefficients'),assets_plan=request.get('assets_plan'),
                first_frame=frame_ids[0],last_frame=frame_ids[-1],host_times_ns=stamps,
                mean_host_ns=float(np.mean(stamps)),source_visit=visit,
                pattern_association='software ACK/time/lease, NOT measured optical synchronization'))
            means.append(mean);single.extend(ff);offsets.append(len(single))
    images=np.stack(means);single=np.asarray(single);features=np.stack([integrate(a,layout) for a in images])
    with (output/'MEANS.npz').open('xb') as f:np.savez_compressed(f,images=images)
    with (output/'FEATURES.npz').open('xb') as f:np.savez_compressed(f,mean_sums=features,single_sums=single,offsets=offsets)
    write(output/'INDEX.json',dict(rows=rows,layout=layout,sources=sources,baseline=ref(ROOT/'BASELINE_01/BASELINE.json'),
        means=ref(output/'MEANS.npz'),features=ref(output/'FEATURES.npz'),source=ref(__file__),
        raw_copied=0,raw_frames=len(single),peak_ADU=peak,clipped_pixel_observations=clipped,
        dataset_roles=list(dict.fromkeys(r['role'] for r in rows)),
        dataset_role='Recorded partition labels; no score/qualification implied',frames_are_not_independent_conditions=True))
    report(output,images,features,single,np.asarray(offsets),rows,layout,peak,clipped)
    print('Audited',len(rows),'queries;',len(single),'RAW; peak',peak,'clipped',clipped,flush=True)


def report(out,images,z,single,offsets,rows,layout,peak,clipped):
    from optical_lab.reporting import plotting,save_figure
    plt=plotting();keys=[r['input_id']+' / '+Path(r['mask']['path']).stem for r in rows]
    names=list(dict.fromkeys(keys));area=np.array([r['pixels'] for r in layout['regions']])
    stats=[]
    for name in names:
        idx=[i for i,key in enumerate(keys) if key==name]
        zz=z[idx];mu=zz.mean(0);sd=zz.std(0,ddof=1) if len(idx)>1 else np.full(64,np.nan)
        stats.append(dict(input=name,queries=len(idx),rms_repeat_ADU=float(np.sqrt(np.mean((sd/area)**2))),
            maximum_region_cv=float(np.nanmax(sd/np.maximum(mu,1e-9))) if len(idx)>1 else None,
            raw_background_in_denominator=True))
    nstats=[]
    for n in (5,10,20,50):
        residual=[];blocks=0;discarded=0
        for i in range(len(rows)):
            ff=single[offsets[i]:offsets[i+1]];count=len(ff)//n
            if count<2:continue
            a=ff[:count*n].reshape(count,n,64).mean(1)/area
            residual.extend((a-a.mean(0))*np.sqrt(count/(count-1)));blocks+=count;discarded+=len(ff)-count*n
        nstats.append(dict(N=n,nonoverlapping_blocks=blocks,discarded_tail_frames=discarded,
            within_query_rms_ADU=float(np.sqrt(np.mean(np.square(residual)))) if residual else None,
            not_independent_cross_visit_evidence=True))
    roles=list(dict.fromkeys(r.get('role','PILOT') for r in rows))
    summary=dict(peak_ADU=peak,clipped_pixel_observations=clipped,repeat_stats=stats,N_comparison=nstats,
        dataset_roles=roles,deployed=False,model_trained=False)
    write(out/'SUMMARY.json',summary)
    cols=min(3,len(names));fig,axes=plt.subplots(2,cols,figsize=(4*cols,7),squeeze=False)
    for j,name in enumerate(names[:cols]):
        idx=[i for i,key in enumerate(keys) if key==name]
        axes[0,j].imshow(images[idx].mean(0),cmap='gray',vmin=8,vmax=25)
        axes[0,j].set_title(name+' | RAW mean / ADU');axes[0,j].axis('off')
        for i in idx:axes[1,j].plot(np.arange(64),z[i]/area,lw=.7)
        axes[1,j].set_xlabel('Region');axes[1,j].set_ylabel('Mean ADU (no subtraction)')
    fig.suptitle('NEW platform '+','.join(roles)+' | fixed display range, not control acceptance')
    fig.tight_layout();save_figure(fig,out/'01_raw_and_regions',180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3))
    for s in stats:ax.bar(s['input'],s['rms_repeat_ADU'])
    ax.tick_params(axis='x',labelrotation=70,labelsize=5)
    ax.set_ylabel('Across-visit region RMS / ADU');fig.tight_layout();save_figure(fig,out/'02_repeatability',180);plt.close(fig)
    with (out/'总结.md').open('x',encoding='utf-8') as f:
        f.write('# 新光路RAW数据审计\n\n数据划分：'+','.join(roles)+'。审计不是模型或部署验收。未使用显示增益、ADU扣除或γ。\n\n')
        f.write(f'完整查询{len(rows)}次，原始单帧{len(single)}张；峰值{peak} ADU，255截顶像素观测数{clipped}。原RAW只索引，不重复复制。\n\n')
        f.write('|输入|查询数|跨次区域均值RMS / ADU|最大RAW区域CV|\n|---|---:|---:|---:|\n')
        for s in stats:f.write(f"|{s['input']}|{s['queries']}|{s['rms_repeat_ADU']:.6f}|{s['maximum_region_cv']}|\n")
        f.write('\nRAW CV的分母包含背景，不能仅凭数值小判定有效信息稳定。N对照使用同次RAW的不重叠子块，仅用于采样选择；不能当作跨次独立验证。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require(Path(a.output).name==a.output and all(Path(b).name==b for b in a.batches), 'explicit direct folders required')
    build(a.batches,ROOT/a.output)
