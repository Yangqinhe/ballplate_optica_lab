"""Offline next-stage design; reuse rendering, NEVER old-platform CCD labels.

Freeze candidate inputs / masks / partitions before observing their NEW-platform
outputs. Four-dimensional trust region first; no automatic expansion to 16D.
"""
import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import numpy as np
from workflow import ROOT,LAB,read,write,ref,require,load
from dataset import checked


def prepare_probes(output,amplitude):
    require(amplitude in (16,24,32),'bounded predeclared diagnostic amplitude')
    output.mkdir();old=read(ROOT/'ASSETS_02/PLAN.json');plan=deepcopy(old)
    base=checked(old['assets']['BASE']['patch'],True);macro=base[::8,::8].astype(float);bases=checked(old['bases'],True)
    assets={'BASE':old['assets']['BASE']}
    for k,axis in enumerate(('X4','Y4','X8','Y8')):
        for sign,label in ((1,'P'),(-1,'M')):
            coeff=np.zeros(4);coeff[k]=sign*amplitude;unbounded=macro+np.einsum('k,khw->hw',coeff,bases)
            gray=np.clip(np.rint(unbounded),0,255).astype(np.uint8).repeat(8,0).repeat(8,1)
            path=output/(axis+label+'.npy')
            with path.open('xb') as f:np.save(f,gray,allow_pickle=False)
            assets[axis+label]=dict(patch=ref(path),coefficients=coeff.tolist(),
                gray_clip_fraction=float(np.mean((unbounded<0)|(unbounded>255))),
                max_gray_delta=int(np.abs(gray.astype(int)-base.astype(int)).max()))
    plan.update(assets=assets,source=ref(__file__),parent=ref(ROOT/'ASSETS_02/PLAN.json'),
        condition_scope='diagnostic stronger raw gray probes; not calibrated radians or trained weights',
        amplitude_raw_gray=amplitude,reason=ref(ROOT/'PILOT_OBSERVABILITY_01/RESULT.json'),
        no_optical_power_or_camera_change=True,hardware_actions=0)
    write(output/'PLAN.json',plan)
    print('OFFLINE stronger probes',amplitude,'gray; no projection',flush=True)


def remaining_catalog(source,output):
    output.mkdir();plan=deepcopy(read(source/'PLAN.json'));catalog=checked(plan['catalogs']['PILOT'])
    smoke={r['input_id'] for r in checked(plan['catalogs']['SMOKE'])['inputs']}
    catalog['inputs']=[r for r in catalog['inputs'] if r['input_id'] not in smoke]
    require(len(catalog['inputs'])==7,'seven remaining inputs required')
    path=output/'REMAINDER_CATALOG.json';write(path,catalog)
    plan['catalogs']['REMAINDER']=ref(path);plan.update(source=ref(__file__),parent=ref(source/'PLAN.json'))
    write(output/'PLAN.json',plan)


def repeated_design(source,output,repeats=3):
    """Preserve frozen pixels/splits; adapt to the original GUI's minimum 3 visits."""
    require(repeats==3,'existing GUI minimum is exactly three visits')
    output.mkdir();plan=deepcopy(read(source/'PLAN.json'))
    for job in plan['jobs']:job['repeats']=repeats
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),
        raw_budget=plan['condition_pairs']*20*repeats,
        sampling_amendment='GUI minimum repeats=3; first N2 batch stopped before any effective frame; pixels/splits/thresholds unchanged',
        previous_failed_attempt=ref(ROOT/'JOINT_01_TRAIN_BASE_00/FAILED.json'))
    write(output/'PLAN.json',plan)


def baseline_readout_design(source,output):
    """Same held inputs under BASE mask for fair later head-only comparison."""
    output.mkdir();parent=read(source/'PLAN.json');plan=deepcopy(parent)
    partitions={s:{} for s in ('SELECT','TEST')}
    for job in parent['jobs']:
        for row in checked(parent['catalogs'][job['catalog']])['inputs']:
            split=row['new_platform_split']
            if split in partitions:partitions[split][row['input_id']]=row
    template=checked(next(iter(parent['catalogs'].values())));catalogs={};jobs=[]
    for split,byid in partitions.items():
        require(len(byid)==8,'eight original held inputs required')
        name=split+'_BASE';cc=deepcopy(template);cc['inputs']=[byid[k] for k in sorted(byid)]
        cc['new_platform_role']=name;path=output/(name+'_CATALOG.json');write(path,cc)
        catalogs[name]=ref(path);jobs.append(dict(catalog=name,mask='BASE',role=name,frames=20,repeats=3,
            input_ids=[r['input_id'] for r in cc['inputs']]))
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),jobs=jobs,catalogs=catalogs,
        condition_pairs=16,raw_budget=960,new_inputs=0,
        purpose='fixed original mask head-only baseline; no new states, masks or numerical thresholds',
        TEST_BASE_remains_unseen_until_model_freeze=True)
    write(output/'PLAN.json',plan)


def time_revisit_design(source,output):
    """Predeclare four known anchors, reverse mask order after model freeze."""
    output.mkdir();plan=deepcopy(read(source/'PLAN.json'));catalog=checked(plan['catalogs']['PILOT'])
    ids=('CENTER','X_POS12','Y_NEG12','VY_POS')
    byid={r['input_id']:r for r in catalog['inputs']};catalog['inputs']=[byid[k] for k in ids]
    catalog['new_platform_role']='TIME_HOLDOUT'
    path=output/'TIME_ANCHORS_CATALOG.json';write(path,catalog)
    plan['catalogs']={'TIME_ANCHORS':ref(path)}
    plan.update(source=ref(__file__),parent=ref(source/'PLAN.json'),
        execution_mask_order=['Y8P','Y8M','X8M','X8P','Y4P','Y4M','X4M','X4P','BASE'],
        sample_N=20,repeats=3,raw_budget=2160,
        time_holdout_requires_model_freeze=True,used_for_training=False,
        unchanged_inputs_and_masks=True,unseen_input_or_mask_NOT_claimed=True)
    write(output/'PLAN.json',plan)


def superposition_design(source,output):
    """Diagnostic tile ablation; not TRAIN/SELECT/TEST or valid control states."""
    output.mkdir();plan=deepcopy(read(source/'PLAN.json'))
    template=checked(plan['catalogs']['TIME_ANCHORS']);original={r['input_id']:r for r in template['inputs']}
    catalogs={};jobs=[]
    for key in ('CENTER','VY_POS'):
        src=original[key];full=checked(src['dmd'],True);encoded=checked(src['encoded_raw96'],True)
        require(full.shape==(600,600) and encoded.shape==(4,1,96,96),'known four-tile layout required')
        inputs=[];tiles=[]
        for name,keep in [('ALL',[0,1,2,3]),('S0',[0]),('S1',[1]),('S2',[2]),('S3',[3]),('OFF',[])]:
            dmd=np.zeros_like(full);small=np.zeros_like(encoded)
            for k in keep:
                y,x=divmod(k,2);dmd[y*300:(y+1)*300,x*300:(x+1)*300]=full[y*300:(y+1)*300,x*300:(x+1)*300]
                small[k]=encoded[k]
            identifier='SP_'+key+'_'+name
            for suffix,array in [('DMD',dmd),('ENCODED',small)]:
                with (output/(identifier+'_'+suffix+'.npy')).open('xb') as f:np.save(f,array,allow_pickle=False)
            row=deepcopy(src);row.update(input_id=identifier,dmd=ref(output/(identifier+'_DMD.npy')),
                encoded_raw96=ref(output/(identifier+'_ENCODED.npy')),new_platform_split='DIAGNOSTIC',
                diagnostic_parent=key,diagnostic_subset=name,retained_tiles=keep,valid_control_state=False)
            row['ball_size']=dict(src['ball_size'],kind='diagnostic_tile_subset',DMD_on_pixels=int(np.count_nonzero(dmd)),
                retained_tiles=keep,control_qualified=False)
            inputs.append(row)
            if len(keep)==1:tiles.append(dmd.astype(np.uint16))
        require(np.array_equal(sum(tiles),full),'tile sum does not reproduce exact original DMD')
        cc=deepcopy(template);cc.update(inputs=inputs,new_platform_role='SUPERPOSITION_DIAGNOSTIC')
        name='SP_'+key;file=output/(name+'_CATALOG.json');write(file,cc);catalogs[name]=ref(file)
        jobs.append(dict(catalog=name,mask='BASE',role='SUPERPOSITION_DIAGNOSTIC',frames=20,repeats=3))
    plan.update(catalogs=catalogs,jobs=jobs,parent=ref(source/'PLAN.json'),source=ref(__file__),
        purpose='diagnostic I(all)-sum I(single)+3 I(DMD_OFF); not coherent-phase identification',
        no_new_SLM=True,no_camera_change=True,not_for_training=True,not_control_states=True,
        DMD_OFF_is_not_physical_dark=True,condition_pairs=12,raw_budget=720)
    write(output/'PLAN.json',plan)


def mixed_training_design(source,output):
    """TRAIN-only non-axis probes; neither SELECT nor TEST pixels are relabeled.

    Axis calibration alone does not constrain mixed-mask interactions. Six
    independent quadratic cross products, each with its opposite, sample the
    same L1<=24 envelope using eight already-declared TRAIN trajectories.
    """
    output.mkdir();parent=read(source/'PLAN.json');plan=deepcopy(parent)
    template=checked(next(iter(parent['catalogs'].values())));byid={}
    for job in parent['jobs']:
        if job['role']!='TRAIN':continue
        for row in checked(parent['catalogs'][job['catalog']])['inputs']:
            require(row['new_platform_split']=='TRAIN','TRAIN-only extension')
            byid[row['input_id']]=row
    ids=['T%02d_Q0'%i for i in (0,3,6,9,12,15,18,19)]
    require(all(k in byid for k in ids),'declared joint TRAIN states missing')
    macro=checked(parent['assets']['BASE']['patch'],True)[::8,::8].astype(float)
    bases=checked(parent['bases'],True);rng=np.random.default_rng(105430)
    candidates=[];cross=[];pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    while len(candidates)<6:
        c=rng.permutation([9.,7.,5.,3.])*rng.choice([-1.,1.],4);c*=np.sign(c[0])
        x=[c[i]*c[j] for i,j in pairs]
        if np.linalg.matrix_rank(np.array(cross+[x]))==len(cross)+1:
            candidates.append(c);cross.append(x)
    assets=deepcopy(parent['assets']);catalogs={};jobs=[]
    oldsha={v['patch']['sha256'] for v in parent['assets'].values()}
    for k,c in enumerate(candidates):
        selected=ids[(k%2)*4:(k%2+1)*4]
        for sign,suffix in ((1,'P'),(-1,'M')):
            name='TRAINMIX%02d%s'%(k,suffix);coef=sign*c
            unbounded=macro+np.einsum('k,khw->hw',coef,bases)
            gray=np.clip(np.rint(unbounded),0,255).astype(np.uint8).repeat(8,0).repeat(8,1)
            path=output/(name+'.npy')
            with path.open('xb') as f:np.save(f,gray,allow_pickle=False)
            require(ref(path)['sha256'] not in oldsha,'TRAIN mask overlaps held mask bytes')
            assets[name]=dict(patch=ref(path),coefficients=coef.tolist(),split='TRAIN',
                gray_clip_fraction=float(np.mean((unbounded<0)|(unbounded>255))),
                max_gray_delta=int(np.abs(gray.astype(int)-macro.repeat(8,0).repeat(8,1)).max()))
            cc=deepcopy(template);cc['inputs']=[byid[i] for i in selected];cc['new_platform_role']='TRAIN_MIX'
            cp=output/(name+'_CATALOG.json');write(cp,cc);catalogs[name]=ref(cp)
            jobs.append(dict(catalog=name,mask=name,role='TRAIN_MIX',frames=20,repeats=3,input_ids=selected))
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),assets=assets,catalogs=catalogs,jobs=jobs,
        condition_pairs=48,raw_budget=2880,new_inputs=0,active_basis_dimensions=4,
        coefficient_L1_max=24,cross_product_rank=int(np.linalg.matrix_rank(cross)),
        reason='SELECT mixed responses failed; add new TRAIN combinations before any formal TEST is opened',
        heldout_inputs_and_masks_unchanged=True,hardware_actions=0)
    write(output/'PLAN.json',plan)


def disentangle_selection_design(source,output):
    """Two held-SELECT contrasts: new input/known mask and known input/new mask."""
    output.mkdir();parent=read(source/'PLAN.json');plan=deepcopy(parent)
    byid={}
    for job in parent['jobs']:
        if job['role'] not in ('TRAIN','SELECT'):continue
        for row in checked(parent['catalogs'][job['catalog']])['inputs']:byid[row['input_id']]=row
    template=checked(next(iter(parent['catalogs'].values())));jobs=[];catalogs={}
    cases=[('SELECT_INPUT_AXIS',['S00_Q0','S00_Q2','S01_Q0','S01_Q2'],['X4P','X4M']),
           ('SELECT_MASK_MIX',['T00_Q0','T03_Q0','T06_Q0','T09_Q0'],['MIX00P','MIX00M'])]
    for role,ids,masks in cases:
        for mask in masks:
            name=role+'_'+mask;cc=deepcopy(template);cc['inputs']=[byid[i] for i in ids];cc['new_platform_role']=role
            path=output/(name+'_CATALOG.json');write(path,cc);catalogs[name]=ref(path)
            jobs.append(dict(catalog=name,mask=mask,role=role,frames=20,repeats=3,input_ids=ids))
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),jobs=jobs,catalogs=catalogs,
        condition_pairs=16,raw_budget=960,new_inputs=0,new_masks=0,
        purpose='separate input and mask generalization within development SELECT; never independent TEST',
        TRAIN_labels_NOT_changed=True,formal_TEST_NOT_opened=True,hardware_actions=0)
    write(output/'PLAN.json',plan)


def complete_training_axes(source,dataset,output):
    """Fill missing signed axes on eight existing TRAIN states; no new TEST use."""
    output.mkdir();parent=read(source/'PLAN.json');plan=deepcopy(parent);idx=read(dataset/'INDEX.json')
    byid={}
    for job in parent['jobs']:
        if job['role']=='TRAIN':
            for row in checked(parent['catalogs'][job['catalog']])['inputs']:
                require(row['new_platform_split']=='TRAIN','TRAIN-only extension');byid[row['input_id']]=row
    ids=['T%02d_Q0'%i for i in (0,3,6,9,12,15,18,19)]
    existing={(r['input_id'],r['mask']['sha256']) for r in idx['rows']}
    template=checked(next(iter(parent['catalogs'].values())));catalogs={};jobs=[]
    for axis in ('X4','Y4','X8','Y8'):
        missing=[]
        for key in ids:
            flags=[(key,parent['assets'][axis+s]['patch']['sha256']) in existing for s in ('P','M')]
            require(flags[0]==flags[1],'unpaired existing TRAIN axis cannot silently duplicate observations')
            if not flags[0]:missing.append(key)
        chunks=[] if not missing else [list(a) for a in np.array_split(missing,int(np.ceil(len(missing)/4)))]
        require(all(2<=len(a)<=4 for a in chunks),'GUI requires at least two inputs per mask batch')
        for chunk_index,selected in enumerate(chunks):
            for sign in ('P','M'):
                mask=axis+sign;name='COMPLETE_'+mask+'_%02d'%chunk_index
                cc=deepcopy(template);cc['inputs']=[byid[k] for k in selected];cc['new_platform_role']='TRAIN_COMPLETE'
                path=output/(name+'_CATALOG.json');write(path,cc);catalogs[name]=ref(path)
                jobs.append(dict(catalog=name,mask=mask,role='TRAIN_COMPLETE',frames=20,repeats=3,input_ids=selected))
    count=sum(len(j['input_ids']) for j in jobs)
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),existing_dataset=ref(dataset/'INDEX.json'),
        catalogs=catalogs,jobs=jobs,condition_pairs=count,raw_budget=count*60,new_inputs=0,new_masks=0,
        purpose='complete all four axis responses on eight existing joint TRAIN inputs; unchanged held partitions',
        actual_camera_and_optics_unchanged=True,formal_TEST_NOT_opened=True)
    write(output/'PLAN.json',plan);print('TRAIN AXES COMPLETION',count,'conditions',len(jobs),'batches',flush=True)


def subset_jobs(source,output,catalog_names):
    """Explicit remaining jobs only; preserve all pixels and data partitions."""
    output.mkdir();plan=deepcopy(read(source/'PLAN.json'))
    require(len(catalog_names)==len(set(catalog_names)),'duplicate explicit job')
    byname={j['catalog']:j for j in plan['jobs']}
    require(set(catalog_names)<=set(byname),'unknown explicit job')
    plan['jobs']=[byname[n] for n in catalog_names]
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),
        raw_budget=sum(len(j['input_ids'])*j['frames']*j['repeats'] for j in plan['jobs']),
        condition_pairs=sum(len(j['input_ids']) for j in plan['jobs']))
    write(output/'PLAN.json',plan)


def merge_test_jobs(source,output):
    """Same sealed eighty TEST cells, grouped by mask to avoid extra exports."""
    output.mkdir();parent=read(source/'PLAN.json');plan=deepcopy(parent);by_mask={};original=set()
    template=checked(next(iter(parent['catalogs'].values())))
    for job in parent['jobs']:
        if job['role'] not in ('TEST_INPUT','TEST_MASK','TEST_BOTH'):continue
        mask=job['mask'];by_mask.setdefault(mask,{})
        for row in checked(parent['catalogs'][job['catalog']])['inputs']:
            key=(row['dmd']['sha256'],parent['assets'][mask]['patch']['sha256'])
            require(key not in original,'duplicate original TEST pixels');original.add(key)
            require(row['input_id'] not in by_mask[mask],'duplicate TEST identity')
            by_mask[mask][row['input_id']]=row
    jobs=[];catalogs={};result=set()
    for mask,rows in by_mask.items():
        require(2<=len(rows)<=8,'merged TEST batch must remain at most eight inputs')
        name='TEST_'+mask;cc=deepcopy(template);cc['inputs']=list(rows.values());cc['new_platform_role']='TEST_MIXED_ROLES'
        path=output/(name+'_CATALOG.json');write(path,cc);catalogs[name]=ref(path)
        jobs.append(dict(catalog=name,mask=mask,role='TEST',frames=20,repeats=3,input_ids=list(rows)))
        result.update((row['dmd']['sha256'],parent['assets'][mask]['patch']['sha256']) for row in rows.values())
    require(result==original and len(result)==80,'TEST coverage changed')
    plan.update(parent=ref(source/'PLAN.json'),source=ref(__file__),catalogs=catalogs,jobs=jobs,condition_pairs=80,
        raw_budget=4800,same_input_mask_SHA_pairs=True,only_export_grouping_changed=True,
        split_defined_per_input_and_mask_not_batch=True,formal_TEST_NOT_yet_captured=True)
    write(output/'PLAN.json',plan);print('TEST GROUPING',len(jobs),'batches; exact',len(result),'conditions',flush=True)


def prepare(output,pilot_assets=ROOT/'ASSETS_02'):
    output.mkdir();pilot=read(pilot_assets/'PLAN.json');amplitude=float(pilot.get('amplitude_raw_gray',8))
    catalog=checked(pilot['catalogs']['PILOT']);encoder=load(LAB/'experiments/E080_ball_input_20261027_1/catalog.py','e105_encoder')
    source=LAB/'experiments/E083_joint_readout_heads_20261028_1/CATALOGS_01/INPUTS.json'
    available=read(source)['inputs'];selected=[]
    for row in available:
        split=row['split'];trajectory=row['trajectory'];q=row.get('query_time_s')
        keep=(split=='train' and (q==0. or (trajectory in ('T00','T01','T02','T03') and abs(q-.24)<1e-9))) or \
             (split=='select' and q in (0.,.24)) or (split=='test' and q==0.)
        if not keep:continue
        r=deepcopy(row);r['new_platform_split']=split.upper();r['optical_label_reused']=False
        raw=checked(r['SOURCE_RAW96'],True);encoded,geometry=encoder.encode_ball_size(raw,2,True)
        require(np.array_equal(encoded,checked(r['encoded_raw96'],True)),'R2 encoding mismatch')
        require(np.array_equal(encoder.pack_frames(encoded),checked(r['dmd'],True)),'DMD encoding mismatch')
        r['ball_size']['geometry']=geometry;encoder.size_info(r,encoded);selected.append(r)
    partitions={s:[r for r in selected if r['new_platform_split']==s] for s in ('TRAIN','SELECT','TEST')}
    require([len(partitions[s]) for s in ('TRAIN','SELECT','TEST')]==[24,8,8],'explicit split sizes differ')
    groups={s:{r['trajectory'] for r in rr} for s,rr in partitions.items()}
    require(not(groups['TRAIN']&groups['SELECT'] or groups['TRAIN']&groups['TEST'] or groups['SELECT']&groups['TEST']),
            'trajectory leakage')
    require(len({r['dmd']['sha256'] for r in selected})==40,'duplicate input bytes')
    for r in selected:require(r['dmd']['sha256'] not in {x['dmd']['sha256'] for x in catalog['inputs']},'pilot pixel overlap')
    assets=deepcopy(pilot['assets']);base=checked(assets['BASE']['patch'],True)[::8,::8].astype(float)
    bases=checked(pilot['bases'],True);rng=np.random.default_rng(105301)
    vectors=np.array([(1,)+v for v in product((-1,1),repeat=3)],float)*(amplitude/4)
    order=rng.permutation(8);mixed={}
    for j,k in enumerate(order):
        split='SELECT' if j<4 else 'TEST'
        for sign,letter in ((1,'P'),(-1,'M')):
            name='MIX%02d%s'%(j,letter);coeff=sign*vectors[k]
            original=base+np.einsum('k,khw->hw',coeff,bases)
            patch=np.clip(np.rint(original),0,255).astype(np.uint8).repeat(8,0).repeat(8,1)
            path=output/(name+'.npy')
            with path.open('xb') as f:np.save(f,patch,allow_pickle=False)
            assets[name]=dict(patch=ref(path),coefficients=coeff.tolist(),split=split,
                gray_clip_fraction=float(np.mean((original<0)|(original>255))),
                max_gray_delta=int(np.abs(patch.astype(int)-checked(assets['BASE']['patch'],True).astype(int)).max()))
            require(assets[name]['max_gray_delta']<=amplitude,'candidate exceeds pixel trust region');mixed[name]=split
    pairs=[]
    # 24 base pairs + 4 directional masks per TRAIN input = 120 TRAIN pairs.
    for j,row in enumerate(partitions['TRAIN']):
        axes=[('X4','Y4','X8','Y8')[j%4],('X4','Y4','X8','Y8')[(j+1)%4]]
        for mask in ['BASE']+[a+s for a in axes for s in ('P','M')]:pairs.append(('TRAIN',row,mask))
    # SELECT: each pair of opposite mixed masks sees the same four inputs.
    sm=[m for m,s in mixed.items() if s=='SELECT']
    tm=[m for m,s in mixed.items() if s=='TEST']
    for j in range(0,8,2):
        for row in partitions['SELECT'][(j//2)%2*4:((j//2)%2+1)*4]:
            for mask in sm[j:j+2]:pairs.append(('SELECT',row,mask))
    # Held inputs: two complete signed directions per input (32 pairs).
    for j,row in enumerate(partitions['TEST']):
        for axis in (('X4','Y4') if j%2==0 else ('X8','Y8')):
            for sign in ('P','M'):pairs.append(('TEST_INPUT',row,axis+sign))
    # Held mixed masks: four existing TRAIN inputs, and two held inputs.
    for j in range(0,8,2):
        for row in partitions['TRAIN'][j:j+4]:
            for mask in tm[j:j+2]:pairs.append(('TEST_MASK',row,mask))
        for row in partitions['TEST'][j:j+2]:
            for mask in tm[j:j+2]:pairs.append(('TEST_BOTH',row,mask))
    require(len(pairs)==232,'sampling budget differs')
    jobs=[];catalogs={}
    # Single-mask batches, maximum eight inputs, two independent N20 visits.
    for role in ('TRAIN','SELECT','TEST_INPUT','TEST_MASK','TEST_BOTH'):
        for mask in dict.fromkeys(m for s,r,m in pairs if s==role):
            rr=[r for s,r,m in pairs if s==role and m==mask]
            for k,start in enumerate(range(0,len(rr),8)):
                name=role+'_'+mask+'_%02d'%k;path=output/(name+'_CATALOG.json')
                cc=dict(schema=catalog['schema'],encoding=catalog['encoding'],inputs=rr[start:start+8],
                    provenance=[ref(source)],new_platform_role=role,old_CCD_truth_used=False)
                write(path,cc);encoder.validate(ref(path));catalogs[name]=ref(path)
                jobs.append(dict(catalog=name,mask=mask,role=role,frames=20,repeats=2,
                    input_ids=[r['input_id'] for r in cc['inputs']]))
    write(output/'PLAN.json',dict(baseline=pilot['baseline'],bases=pilot['bases'],assets=assets,catalogs=catalogs,jobs=jobs,
        rendering_source=ref(source),source=ref(__file__),new_inputs=40,unique_trajectories={s:len(g) for s,g in groups.items()},
        new_platform_labels_not_yet_collected=True,global_historical_pixel_novelty_NOT_claimed=True,
        physical_camera_or_GPU_actions=0,raw_budget=232*2*20,condition_pairs=232,
        active_basis_dimensions=4,not_16_dimensions=True,coefficient_L1_max=amplitude,max_pixel_delta_gray=amplitude,
        pilot_assets=ref(pilot_assets/'PLAN.json'),
        no_execution_authorized_by_this_file=True,requires_pilot_observability=True))
    print('OFFLINE design:',len(jobs),'batches, 232 pairs, 9280 RAW; 4D local envelope only',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='DESIGN_01');p.add_argument('--pilot-assets',default='ASSETS_02');a=p.parse_args()
    require(Path(a.output).name==a.output and Path(a.pilot_assets).name==a.pilot_assets,'direct new output required')
    prepare(ROOT/a.output,ROOT/a.pilot_assets)
