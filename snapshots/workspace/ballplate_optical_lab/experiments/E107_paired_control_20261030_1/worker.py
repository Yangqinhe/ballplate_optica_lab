"""Separate MuJoCo/model process. Never imports camera SDK or display clients."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import core as c


class Pair:
    def __init__(self,package,output,device=None):
        self.package=package; self.output=Path(output).resolve(); self.output.mkdir(exist_ok=False)
        for source in package['sources']: c.checked(source)
        self.heads={k:c.read(c.checked(v)) for k,v in package['heads'].items()}
        self.mask=np.load(c.checked(package['mask'])); self.layout=package['layout']
        for h in self.heads.values():
            if h['layout']!=self.layout or h['mask']!=package['mask']: raise ValueError('head contract mismatch')
        os.environ.setdefault('MUJOCO_GL','egl')
        # Project-local OpenCV, no changes to the running GUI environment.
        sys.path.insert(0,str(c.ROOT/'deps'))
        import cv2,mujoco
        cv2.setNumThreads(2); self.cv2=cv2; self.mj=mujoco
        self.reference=c.load(c.EXP/'E098_model_closedloop_20261029_1/evaluate.py','e107_reference_physics')
        self.encoder=c.load(c.EXP/'E080_ball_input_20261027_1/catalog.py','e107_encoder')
        camera=c.load(self.reference.CAMERA,'e107_rendered_camera')
        self.model=mujoco.MjModel.from_xml_path(str(self.reference.MODEL))
        self.camera=camera.BallCamera(self.model,128,128); self.binary=camera.binary_ball
        self.ball=int(self.model.jnt_qposadr[self.model.joint('ball_free').id])
        self.dof=int(self.model.jnt_dofadr[self.model.joint('ball_free').id])
        self.mass=float(self.model.body_mass[self.model.body('ball').id])
        self.sensor=int(self.model.sensor_adr[self.model.sensor('ball_position').id])
        self.plate=[int(self.model.sensor_adr[self.model.sensor(k).id]) for k in ('plate_roll_angle','plate_pitch_angle')]
        self.act=[self.model.actuator(k).id for k in ('roll_target','pitch_target')]
        self.preview=c.load(c.PREVIEW/'model.py','e107_surrogate')
        self.manifest=c.read(c.checked(package['model'])); self.engine=self.preview.Engine(self.manifest,device)
        self.episode=-1; self.pending=None; self.reset([0.,0.])
        c.write(self.output/'PROTOCOL.json',dict(package=package,physics_dt=float(self.model.opt.timestep),
            control_dt=c.DT,ball_mass_kg=self.mass,actual_hardware=False,
            notes='hardware evidence only exists in step records with real_RAW archives',
            optical_history_s=[.18,.12,.06,0],sensor_readout='RAW, not display enhanced',
            reset='identical position, zero velocity, repeated initial rendered history',
            impulse='N s at ball COM in world XY, queued after observation and before next physics step',
            online_truth_feedback=False,SLM_fixed=True,closed_loop_qualified=False))

    def reset(self,xy):
        xy=np.asarray(xy,float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or np.linalg.norm(xy)>.201:
            raise ValueError('复位位置须在20cm半径内')
        self.episode+=1; self.step=0; self.pending=None; self.worlds=[]
        for _ in range(2):
            d=self.mj.MjData(self.model); self.mj.mj_resetData(self.model,d)
            d.qpos[self.ball:self.ball+2]=xy; self.mj.mj_forward(self.model,d)
            self.worlds.append(dict(data=d,history=c.History(),limited=np.zeros(2),filtered=np.zeros(2),
                cached_step=-1,dmd=None))
        return self.states()

    def states(self):
        result=[]
        for w in self.worlds:
            d=w['data']; p=d.sensordata[self.sensor:self.sensor+3].copy()
            result.append(dict(time=float(d.time),position=p.tolist(),velocity_world=d.qvel[self.dof:self.dof+2].tolist(),
                angles=d.sensordata[self.plate].tolist(),filtered_action=w['filtered'].tolist(),
                fallen=bool(np.max(np.abs(p[:2]))>.27 or p[2]<-.05)))
        return result

    def dmd(self,w):
        if w['cached_step']!=self.step:
            rgb=self.camera.capture(w['data']); binary=self.binary(rgb)
            a=self.reference.resize_binary(binary[None])
            history=w['history'].push(a)
            encoded,_=self.encoder.encode_ball_size(history,self.package['radius'],True)
            w['dmd']=self.encoder.pack_frames(encoded); w['cached_step']=self.step
        return w['dmd']

    def prepare(self,token,context):
        if self.pending is not None: raise ValueError('previous observation not consumed/cancelled')
        self.preview.validate_context(context,self.manifest)
        ds=[self.dmd(w) for w in self.worlds]
        predicted,metadata=self.engine.predict(ds[0],self.mask,context)
        z=c.integrate(predicted['corrected'],self.layout)
        folder=self.output/f'E{self.episode:03d}_T{self.step:06d}_Q{token:06d}'; folder.mkdir()
        archive=folder/'observation.npz'
        with archive.open('xb') as f:
            np.savez(f,simulation_DMD=ds[0],real_DMD=ds[1],SLM=self.mask,
                simulation_CCD=predicted['corrected'],simulation_sums=z)
        self.pending=dict(token=token,folder=folder,z=z,states=self.states(),context=context,metadata=metadata)
        return dict(archive=str(archive),states=self.states(),episode=self.episode,step=self.step,model=metadata)

    def advance(self,observation_id,archive,impulse,model_only=False):
        p=self.pending
        if p is None or observation_id!=p['token']: raise ValueError('过期/重复观测，不推进MuJoCo')
        archive=Path(archive).resolve()
        if archive.parent!=p['folder']: raise ValueError('capture outside current observation folder')
        with np.load(archive,allow_pickle=False) as a:
            raw=a['raw']; measured=a['image']
            if not model_only and (raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[1:]!=tuple(self.layout['shape'])):
                raise ValueError('real RAW evidence required')
        zr=c.integrate(measured,self.layout)
        actions=[c.action(p['z'],self.heads['simulation']),c.action(zr,self.heads['real'])]
        dv=c.impulse_velocity(impulse[0],impulse[1],self.mass)
        if any(s['fallen'] for s in self.states()): raise ValueError('球已离板，请复位后继续')
        for w,u in zip(self.worlds,actions):
            w['limited'],w['filtered']=self.reference.limit_filter(u,w['limited'],w['filtered'])
            d=w['data'];d.qvel[self.dof:self.dof+2]+=dv;d.ctrl[self.act]=w['filtered']
            self.mj.mj_step(self.model,d,nstep=round(c.DT/self.model.opt.timestep))
        self.step+=1
        record=dict(episode=self.episode,step=self.step,states_before=p['states'],states_after=self.states(),
            simulation_sums=p['z'].tolist(),real_sums=zr.tolist(),raw_actions=np.asarray(actions).tolist(),
            impulse_Ns=impulse[0],impulse_direction_deg=impulse[1],delta_v_world=dv.tolist(),
            real_capture=c.ref(archive),camera_context=p['context'],model=p['metadata'],
            hardware_CCD_used=not model_only,hardware_qualified=False,simulated_step_s=c.DT)
        c.write(p['folder']/'STEP.json',record);self.pending=None
        return record

    def close(self): self.camera.close()


def serve(package_ref,output):
    pair=None
    for line in sys.stdin:
        q={}
        try:
            q=json.loads(line); op=q['op']
            if op=='init':
                if pair is not None: raise ValueError('already initialized')
                pair=Pair(c.read(c.checked(package_ref)),output)
                result=dict(states=pair.states(),mass=pair.mass)
            elif pair is None: raise ValueError('init required')
            elif op=='reset': result=dict(states=pair.reset(q['xy']))
            elif op=='prepare': result=pair.prepare(q['id'],q['context'])
            elif op=='advance': result=pair.advance(q['observation_id'],q['archive'],q['impulse'])
            elif op=='cancel': pair.pending=None;result={}
            elif op=='close': pair.close();return
            else: raise ValueError('unknown command')
            reply=dict(ok=True,id=q['id'],op=op,result=result)
        except Exception as e: reply=dict(ok=False,id=q.get('id'),op=q.get('op'),error=str(e))
        print('E107 '+json.dumps(reply,ensure_ascii=False,allow_nan=False),flush=True)
    if pair is not None:pair.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();serve(dict(path=str(a.package),sha256=a.sha256),a.output)
