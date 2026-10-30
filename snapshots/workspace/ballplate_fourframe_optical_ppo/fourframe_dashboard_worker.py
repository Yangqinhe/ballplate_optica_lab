"""One owner for MuJoCo/CUDA, explicit disturbance and ablation commands."""
import fourframe_runtime as rt
import queue
import time
import traceback
import json
from dashboard.worker import DashboardWorker
from fourframe_dashboard_session import FourFrameSession,save_fourframe_bundle,DEFAULT_FLAT


class FourFrameWorker(DashboardWorker):
    def __init__(self,*args,flat_checkpoint=DEFAULT_FLAT,**kwargs):
        super().__init__(*args,**kwargs)
        self.arguments["flat_checkpoint"]=flat_checkpoint

    def run(self):
        session=None;trace=events=None;created=False;started=time.perf_counter()
        try:
            self.output.mkdir(parents=True,exist_ok=False);created=True
            session=FourFrameSession(**self.arguments)
            self.metadata=session.metadata();rt.json_save(self.output/"metadata.json",self.metadata)
            trace=(self.output/"control_trace.jsonl").open("x")
            events=(self.output/"events.jsonl").open("x")
            self.running=not self.start_paused
            self.publish("ready",metadata=self.metadata,output=str(self.output));self.publish_snapshot(session.snapshot())
            while not self.stop_requested.is_set():
                try:command,payload=self.commands.get(timeout=.01 if self.running else .1)
                except queue.Empty:command,payload="idle",{}
                if command not in ("idle","wake"):
                    event={"command":command,**payload,"active_variant_before":session.variant,
                        "frame_id":session.current_ccd.frame_id,"simulation_time_s":session.simulation_time,
                        "received_unix_s":time.time()}
                    events.write(json.dumps(event,ensure_ascii=False)+"\n");events.flush()
                advance=False
                if command=="run":self.running=not session.finished
                elif command=="pause":self.running=False
                elif command=="step":self.running=False;advance=not session.finished
                elif command=="reset":
                    self.running=False;session.reset(payload.get("xy") if "xy" in payload else None if payload.get("random") else [0.,0.])
                elif command=="push" and not session.finished:session.push(payload["direction"],payload["force"],payload["duration"])
                elif command=="kick" and not session.finished:session.kick(payload["direction"],payload["speed"])
                elif command=="release":session.release()
                elif command=="variant":
                    self.running=False;session.set_variant(payload["variant"])
                    self.metadata=session.metadata();rt.json_save(self.output/"metadata.json",self.metadata)
                    self.publish("ready",metadata=self.metadata,output=str(self.output))
                elif command=="save":
                    path=self.output/f"bundle_{session.current_ccd.frame_id:08d}_{time.time_ns()}"
                    save_fourframe_bundle(session.snapshot(),path,self.metadata);self.publish("saved",path=str(path))
                if self.stop_requested.is_set():break
                if self.running or advance:
                    before=session.global_steps;snapshot=session.step()
                    if session.global_steps>before:
                        trace.write(json.dumps(session.last_control_row)+"\n");trace.flush()
                    if session.finished:self.running=False
                    self.publish_snapshot(snapshot)
                elif command not in ("idle","wake","save"):self.publish_snapshot(session.snapshot())
        except Exception:
            self.error=traceback.format_exc();print(self.error,flush=True);self.publish("error",error=self.error)
        finally:
            if trace:trace.close()
            if events:events.close()
            report=dict(role="fourframe_simulation_dashboard",hardware_enabled=False,end_reason=self.close_reason,
                wall_seconds=time.perf_counter()-started,error=self.error)
            if session is not None:
                try:
                    save_fourframe_bundle(session.snapshot(),self.output/"final_bundle",session.metadata())
                    report.update(global_steps=session.global_steps,last_frame_id=session.current_ccd.frame_id,
                        simulation_time_s=session.simulation_time,displayed_native_readout_error=session.current_ccd.cache_max_error,
                        optical_forward_calls=session.optical_forward_calls,ccd_frames_retained=len(session.records),variant=session.variant)
                    session.close();report["weights_unchanged"]=True
                except Exception:self.error=traceback.format_exc();report["close_error"]=self.error
            if created:rt.json_save(self.output/"report.json",report)
            self.running=False;self.publish("closed",report=report)
