"""Prospective manual-standby black-only startup/long-capture qualification.

Starts ONLY the previously accepted v101 volatile sequence, while original
windows stay actually all-zero. No definition/mode/crop/trigger-delay writes.
First frame counts; no warmup discard or failed-image retry. Stops sequence
on completion/failure and restores the NEW approved manual camera baseline.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import sys
import time

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'tools'),str(PROJECT/'live_manual_v17')]
from configure_singlebit_vsync_v101 import write_packet,validate_ack
from read_dlpc900_linux_v100 import codec,discover,read_packet
from diagnose_black_restart_v83 import TimedBlackHold,wait_black,telemetry
from probe_line0_trigger_v18 import BlackHold
from qualify_singlebit_ccd_v96 import ROI,timing_summary
from diagnose_settled_trigger_writes_v88 import uncached_contract
from live_manual_v19_trigger_r8.camera import SettledStandbyCamera
from live_manual_v19_trigger.camera import validate_contract
from inspect_recovery_ui_v20 import inspect
from camera_recheck import verify_baseline
from live_adapt_v22.core import digest,write_new


def validate_target(evidence):
    target=evidence['baseline']
    check=evidence.get('verification',{})
    if (evidence.get('schema')!='USER_APPROVED_NEW_MANUAL_STANDBY_V107'
            or not evidence.get('approved_by_user') or not check.get('passed')
            or check.get('target')!=target or len(check.get('snapshots',[]))!=10
            or target.get('DeviceSerialNumber')!='15328133'
            or target.get('ExposureAuto')!='Off' or target.get('GainAuto')!='Off'
            or target.get('TriggerMode')!='Off' or float(target['Gain'])!=0.
            or not 950<=float(target['ExposureTime'])<=1050):
        raise ValueError('strictly verified approved new manual standby target required')
    return target


def frame_record(raw,meta):
    return dict(**meta,min_ADU=int(raw.min()),max_ADU=int(raw.max()),
        mean_ADU=float(raw.mean()),saturated_pixels=int(np.count_nonzero(raw>=240)))


def validate_frame(raw,record,previous):
    if raw.dtype!=np.uint8 or raw.shape!=(554,552) or record['max_ADU']>=240:
        raise ValueError('first-or-later complete raw frame format/ROI/headroom failed')
    if previous and any(record[k]<=previous[k] for k in ('FrameID','TimeStamp','host_received_unix_ns')):
        raise ValueError('non-monotonic delivered frame')


class Controller:
    def __init__(self,guard,folder,profile_result):
        self.guard,self.folder=guard,folder
        self.protocol=codec();self.devices=discover();self.fd=self.lock=None
        self.rows=[];self.started_by_us=False;self.stopped_verified=False
        self.source_bindings={str(PROJECT/'tools'/n):digest(PROJECT/'tools'/n)
            for n in ('configure_singlebit_vsync_v101.py','read_dlpc900_linux_v100.py',
                      'qualify_manual_black_v108.py')}
        self.profile_result=profile_result;self.profile_hash=digest(profile_result)

    def open(self):
        if len(self.devices)!=1 or not self.devices[0]['user_can_read_write']:
            raise RuntimeError('unique permitted DLPC900 interface 0 required')
        profile=json.loads(self.profile_result.read_text())
        if (not profile.get('completed') or not profile.get('candidate_retained')
                or not profile.get('health',{}).get('status_reads_passed')):
            raise ValueError('documented accepted v101 profile required; no unknown redefinition')
        self.lock=open(f'/run/user/{os.getuid()}/ballplate_dlpc900_readonly.lock','a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.fd=os.open(self.devices[0]['node'],os.O_RDWR|os.O_NONBLOCK|os.O_CLOEXEC)
        discarded=[]
        for _ in range(16):
            if not select.select([self.fd],[],[],0)[0]:break
            discarded.append(os.read(self.fd,65).hex())
        write_new(self.folder/'DISCARDED_PREEXISTING_USB.json',discarded)

    def transaction(self,name,run=None,cleanup=False):
        if cleanup:
            if run is not False:raise ValueError('cleanup permits STOP only')
            BlackHold.check(self.guard,force=True)
        else:self.guard.check(force=True)
        if (discover()!=self.devices or digest(self.profile_result)!=self.profile_hash
                or any(digest(p)!=sha for p,sha in self.source_bindings.items())):
            raise RuntimeError('controller/evidence/source identity changed')
        seq=len(self.rows)+1
        if seq>255:raise RuntimeError('USB sequence budget exceeded')
        packet=(read_packet(self.protocol,name,seq) if run is None
                else write_packet(0x1A24,b'\x02' if run else b'\0',seq))
        row=dict(name=name,operation='READ' if run is None else 'START' if run else 'STOP',
                 seq=seq,request_hex=packet.hex(),unix_ns=time.time_ns(),cleanup=cleanup)
        self.rows.append(row);write_new(self.folder/f'USB_{seq:03d}_INTENT.json',row)
        try:
            if run:self.started_by_us=True
            if os.write(self.fd,packet)!=65:raise RuntimeError('short HID request')
            row['sent']=True
            if not select.select([self.fd],[],[],1.5)[0]:raise TimeoutError('USB reply deadline; no retry')
            raw=os.read(self.fd,65);row['reply_hex']=raw.hex()
            row['decoded']=(self.protocol.decode_reply(name,self.protocol.parse_reply(name,seq,raw))
                if run is None else validate_ack(raw,seq))
            return row['decoded']
        except BaseException as error:row['error']=repr(error);raise
        finally:write_new(self.folder/f'USB_{seq:03d}_RESULT.json',row)

    def start(self):
        if (self.transaction('display_mode')['mode']!='video_pattern'
                or self.transaction('pattern_invert')['inverted']
                or self.transaction('pattern_lut_config')!=dict(entries=1,patterns_to_display=0,repeat_indefinitely=True)):
            raise RuntimeError('unexpected current mode/invert/LUT; no changes attempted')
        before=self.transaction('main_status')
        if before['sequencer_running'] or before['parked'] or before['video_frozen'] or not before['external_video_locked']:
            raise RuntimeError('expected stopped, locked, unparked and unfrozen sequence')
        self.transaction('start_existing_known_sequence',run=True)
        pairs=[]
        for _ in range(4):
            h=self.transaction('hardware_status');m=self.transaction('main_status')
            pairs.extend([dict(name='hardware_status',decoded=h),dict(name='main_status',decoded=m)])
            time.sleep(.15)
        health=self.protocol.assess_status_series(pairs)
        write_new(self.folder/'CONTROLLER_START_HEALTH.json',health)
        if not health['status_reads_passed']:raise RuntimeError('controller health failed after start')

    def close(self):
        errors=[]
        if self.fd is not None and self.started_by_us:
            try:
                self.transaction('stop_after_black_cohort',run=False,cleanup=True)
                # One read only while normal deadline remains; STOP is still
                # allowed on expiry, but never pretend an unverified stop passed.
                status=self.transaction('main_status')
                self.stopped_verified=not status['sequencer_running']
                if not self.stopped_verified:raise RuntimeError('sequence still running after STOP')
            except BaseException as error:errors.append(repr(error))
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.lock is not None:self.lock.close();self.lock=None
        return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--manual-target',type=Path,required=True)
    p.add_argument('--profile-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();target=validate_target(json.loads(a.manual_target.read_text()))
    counts=[31]*6+[3000]*2
    plan=dict(source_sha256=digest(__file__),manual_target_sha256=digest(a.manual_target),
        profile_result_sha256=digest(a.profile_result),counts=counts,maximum_seconds=300,
        requested_exposure_us=1000.,gain=0.,ROI=ROI,trigger='Line0/FallingEdge/Rolling',
        first_frame_included=True,warmup_discards=0,failed_frame_retries=0,
        zero_only=True,only_controller_writes='START and STOP previously accepted sequence',
        prior_failed_cohorts_unchanged=True,no_new_windows_no_GUI_edits=True,release_approved=False)
    if not a.execute:print(json.dumps(plan,indent=2));return
    a.output.mkdir(parents=True,exist_ok=False);write_new(a.output/'PROSPECTIVE_COHORT.json',plan)
    preflight=inspect(a.session,parked=True);write_new(a.output/'ORIGINAL_BLACK_PREFLIGHT.json',preflight)
    started=time.monotonic();guard=TimedBlackHold(a.session,preflight,started+300)
    controller=Controller(guard,a.output,a.profile_result)
    result=dict(passed=False,error=None,rows=[],release_approved=False)
    camera=None
    try:
        controller.open();controller.start()
        for index,count in enumerate(counts):
            guard.check(force=True);folder=a.output/f'trial_{index:02d}_{count}';folder.mkdir()
            row=dict(index=index,requested=count,frames=[],passed=False,restored=False,error=None)
            try:
                camera=SettledStandbyCamera(guard)
                if camera.before!=target:raise RuntimeError('new manual standby target changed')
                camera.set_roi(ROI);camera.start_source(1000.,'Line0')
                row['active']=uncached_contract(camera);validate_contract(row['active'],1000.,ROI,'Line0')
                row['initial_telemetry']=telemetry(camera)
                end=time.monotonic()+(80 if count>31 else 4)
                for j in range(count):
                    guard.check()
                    if time.monotonic()>=end:raise TimeoutError('predeclared acquisition deadline')
                    raw,meta=camera.capture(timeout_ms=1000);record=frame_record(raw,meta)
                    try:validate_frame(raw,record,row['frames'][-1] if row['frames'] else None)
                    except Exception:
                        np.save(folder/'REJECTED_raw.npy',raw,allow_pickle=False)
                        write_new(folder/'REJECTED_METADATA.json',record);raise
                    row['frames'].append(record)
                    if j<3 or j==count-1:np.save(folder/f'raw_{j:04d}.npy',raw,allow_pickle=False)
                    if count>31 and (j+1)%1000==0:
                        print(json.dumps(dict(index=index,received=j+1)),flush=True)
                row['timing']=timing_summary(row['frames'])
                row['final_telemetry']=telemetry(camera)
                validate_contract(uncached_contract(camera),1000.,ROI,'Line0')
                row['passed']=True
            except BaseException as error:row['error']=repr(error)
            finally:
                if camera is not None:
                    try:camera.close();row['restored']=camera.after==target
                    except BaseException as error:row['close_error']=repr(error)
                    row['after']=getattr(camera,'after',None);camera=None
                write_new(folder/'TRIAL_RESULT.json',row);result['rows'].append(row)
                print(json.dumps(dict(index=index,requested=count,received=len(row['frames']),
                    passed=row['passed'],restored=row['restored'],error=row['error'],
                    close_error=row.get('close_error'))),flush=True)
            if not row['passed'] or not row['restored']:
                raise RuntimeError('first failed capture or restore stops this NEW cohort; no retry')
            wait_black(guard,.2)
        result['all_captures_passed']=True
    except BaseException as error:result['error']=repr(error)
    finally:
        if camera is not None:
            try:camera.close()
            except BaseException as error:result['last_close_error']=repr(error)
        result['controller_cleanup_errors']=controller.close()
        result['controller_stopped_verified']=controller.stopped_verified
        result['controller_transactions']=controller.rows
        try:
            result['final_manual_baseline_check']=verify_baseline(target,lambda:guard.check(force=True))
            result['final_black']=inspect(a.session,parked=True)
        except BaseException as error:result['final_check_error']=repr(error)
        guard.close()
        result.update(wall_seconds=time.monotonic()-started,
            passed=bool(result.get('all_captures_passed') and controller.stopped_verified
                and not result['controller_cleanup_errors'] and 'final_black' in result
                and result.get('final_manual_baseline_check',{}).get('passed')),
            hardware_20ms_qualified=False,physical_black_mapping_qualified=False,
            optical_integration_window_qualified=False,no_new_projection=True,no_new_windows=True)
        write_new(a.output/'MANUAL_BLACK_QUALIFICATION_RESULT.json',result)
        print(json.dumps(dict(passed=result['passed'],error=result['error'],
            passed_trials=sum(r['passed'] for r in result['rows']),trials=len(result['rows']),
            stopped=controller.stopped_verified,final_manual_baseline=result.get('final_manual_baseline_check',{}).get('passed'))),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
