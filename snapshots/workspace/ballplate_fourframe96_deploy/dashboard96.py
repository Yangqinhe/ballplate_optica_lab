"""Tk dashboard for the exact frozen 96x4/C150 candidate; simulation only."""
import runtime96 as rt
import argparse
from datetime import datetime
import os
from pathlib import Path
import tkinter as tk
from PIL import Image,ImageDraw,ImageFont,ImageTk
import numpy as np
import fourframe_dashboard as ui
import fourframe_dashboard_worker as worker_module
from dashboard.widgets import CjkLabel,CjkButton,FONT_PATH,PANEL,BACKGROUND,MUTED,ACCENT,WARNING
from dashboard.session import full_panel
from readout_overlay import render_ccd,REGION_COLORS
from dashboard96_session import Session96,save96,DEFAULT_CHECKPOINT,VARIANTS,REGIONS,statistics96,matching_readout96
from hardware_contract import slm_preview


# Explicit process-local adapters only; no old source or old GUI is modified.
ui.VARIANTS=VARIANTS
worker_module.FourFrameSession=Session96
worker_module.save_fourframe_bundle=save96


def render96(counts,gain,size,regions=True):
    # Shared painter expects raw180 orientation. Convert ONLY a display copy
    # and request its corrected view, preserving actual registered CCD data.
    rendered=render_ccd(np.ascontiguousarray(counts[::-1,::-1]),REGIONS,gain=gain,size=size,
        show_regions=regions,labels=True,corrected_view=True)
    if regions and size<400:
        preview=Image.fromarray(rendered);draw=ImageDraw.Draw(preview)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
        for i,(y0,y1,x0,x1) in enumerate(REGIONS):
            draw.text((x0*size/600+3,y0*size/600+3),f'R{i+1}',font=font,fill=REGION_COLORS[i],stroke_width=1,stroke_fill='#08101b')
        rendered=np.asarray(preview)
    return rendered


class App96(ui.FourFrameApp):
    def __init__(self,*args,**kwargs):
        self.final_gray_preview=False
        super().__init__(*args,**kwargs)
        self.preview_gain=16.
        self.root.title('96×96 四幅 · C150 / SLM4×4 · 手动扰动 · 全仿真（非实机）')
    def build(self):
        super().build()
        outer=self.root.winfo_children()[0]
        labels=[w for w in outer.winfo_children() if isinstance(w,CjkLabel)]
        labels[0].set('96×96 四幅 · C150 / SLM 4×4 · 手动抗扰动')
        labels[1].set('4×96² → 192²拼图 / DMD600 → 单SLM一次传播 → 当前注册CCD600八区 → 18参数线性头')
        self.rgb.caption.set('MuJoCo诊断视角；策略仅看缩小后的四幅96²二值图')
        self.dmd.title.set('02 DMD · 四幅96²')
        self.slm.title.set('03 SLM · C150 / 600²')
        self.ccd_panel.title.set('04 当前模拟CCD · 注册600²')
        self.corner_button.set('角落复位')
        self.phase_view_button=CjkButton(self.monitor_column,'SLM指令 / LUT后预览 L',command=self.toggle_phase_view)
        self.phase_view_button.pack(anchor='w',pady=4)
    def toggle_phase_view(self):
        self.final_gray_preview=not self.final_gray_preview
        if self.latest:self.draw(self.latest)
    def toggle_gain(self):
        self.preview_gain={1.:4.,4.:16.,16.:64.,64.:1.}[self.preview_gain]
        if self.latest:self.draw(self.latest)
    def key(self,event):
        if event.keysym.lower()=='l':self.toggle_phase_view();return 'break'
        return super().key(event)
    def draw(self,s):
        self.paint_count+=1;self.rgb.set_array(s.rgb)
        if self.native_panels:
            panel=np.zeros((600,800),np.uint8);panel[:,100:700]=s.dmd_active
            self.dmd.set_array(panel,outline=(100,0,600,600))
            dmd_note='历史800×600视频示意；不输出器件'
        else:
            image=Image.fromarray(s.dmd_active).convert('RGB');draw=ImageDraw.Draw(image)
            font=ImageFont.truetype(str(FONT_PATH),23)
            for i,(age,color) in enumerate(zip((60,40,20,0),REGION_COLORS[:4])):
                x,y=12+(i%2)*288,12+(i//2)*288
                draw.rectangle((x,y,x+287,y+287),outline=color,width=3)
                draw.text((x+7,y+6),f"{'当前' if age==0 else f'-{age}ms'} #{s.raw_input_ids[i]}",font=font,fill=color)
            self.dmd.set_array(np.asarray(image));dmd_note='192²逻辑 → 576²内容 + 12px黑边'
        fill='冷启动重复填充；' if len(set(s.raw_input_ids))<4 else ''
        self.dmd.caption.set(f'{dmd_note}\n600²画布；{fill}ID {"/".join(map(str,s.raw_input_ids))}；标注仅显示')
        phase=slm_preview(s.slm_active) if self.final_gray_preview else s.slm_active
        phase=ui.PHASE_COLORS[phase] if self.phase_color else phase
        if self.native_panels:
            if phase.ndim==2:phase=np.repeat(phase[...,None],3,axis=2)
            canvas=np.zeros((1080,1920,3),np.uint8);canvas[240:840,660:1260]=phase;phase=canvas
        self.slm.set_array(phase)
        phase_kind='历史LUT后+水平翻转预览' if self.final_gray_preview else '规范相位指令'
        self.slm.caption.set(f'{phase_kind}；{"伪彩" if self.phase_color else "灰度"}仅显示\nC150：4×4=16个像素/控制点；600²\n指令{s.phase_gray_range[0]}–{s.phase_gray_range[1]}，{s.phase_levels}级；非实机标定')
        numeric=self.metadata.get('known_main_model_sampling_action_difference_rad')
        warning='采样差未知，未验收' if numeric is None else f'采样差{numeric*1000:.1f}mrad > 2mrad，禁止据此实机放行'
        self.banner.set(f'全仿真 · {VARIANTS[s.variant]} · {warning}')
        ended={'missing_optical_signal':'光学信号为空，已停止','fell':'越界/掉球，请复位','episode_complete':'演示回合完成，请复位'}
        status=ended.get(s.end_reason,'回合结束，请复位') if s.finished else '闭环运行' if self.running else '已暂停'
        self.state_label.set(f'{status}\n回合{s.episode} / t={s.simulated_time_s:.2f}s\n当前观测#{s.frame_id}；仅推理，不训练')
        self.pause_button.set('请复位' if s.finished else '暂停（空格）' if self.running else '开始（空格）')
        for button in [self.pause_button,self.step_button,*self.push_buttons,*self.kick_buttons]:button.configure(state='disabled' if s.finished else 'normal')
        frame=s.history[0].frame;stats=statistics96(frame.raw_counts,self.metadata['ccd_readout']['reference_counts'])
        if not matching_readout96(stats['features'],frame.features):raise RuntimeError('UI/actor current8 mismatch')
        self.ccd_panel.set_array(render96(frame.raw_counts,self.preview_gain,320,self.show_readout_regions))
        self.ccd_panel.caption.set(f'注册600²模拟Mono8，不是原始传感器ROI\n#{s.frame_id}；{self.preview_gain:g}×仅显示；点击放大')
        self.readout_label.set(f'q = 当前8区计数 ÷ {self.metadata["ccd_readout"]["reference_counts"]:g}\n显示/策略误差{s.cache_error:.1g}；仅当前一张CCD')
        self.flux_chart.delete('all');q=frame.features;limit=max(float(q.max()),1e-8)
        for i,(count,value) in enumerate(zip(stats['region_counts'],q)):
            y=10+i*27;color=REGION_COLORS[i]
            self.flux_chart.create_rectangle(30,y+2,30+max(1,float(value)/limit*108),y+15,fill=color,outline='')
            self.flux_chart.create_text(2,y+8,anchor='w',text=f'R{i+1}',fill=color)
            self.flux_chart.create_text(148,y+8,anchor='w',text=f'q={value:.5f} Σ={int(count)}',fill='#e7eff8',font=('TkFixedFont',10))
        self.readout_note.set('八区覆盖注册600²，合计360,000数字像素。\n没有CCD历史；无图像质心/真实状态输入。\n置零相位只是同头消融，不是重训公平基线。')
        self.sync.set(f'RAW96缓存4幅（60/40/20/0ms） → 单次光路 → 当前CCD8 → Linear18｜累计传播{s.optical_forward_calls}次')
        pending,applied=s.next_mean_action,s.last_applied_action
        self.control_label.set(f'当前CCD → 下一均值(rad)\nroll {pending[0]:+.5f}  pitch {pending[1]:+.5f}\n上一周期执行（限幅/低通后）\nroll {applied[0]:+.5f}  pitch {applied[1]:+.5f}')
        self.diagnostic_label.set(f'诊断 x={s.position[0]*100:+.2f} y={s.position[1]*100:+.2f} cm\n速度{s.speed*100:.2f}cm/s；稳定{s.stable}\n奖励{s.reward_sum:.1f}；外力剩{s.force_steps_remaining}步\n位置/速度不输入策略')
        self.performance_label.set(f'ASM {s.grid_size} / padding后{s.grid_size*3} / {s.device}\n采集{s.capture_seconds*1000:.0f}ms；均步{s.mean_step_seconds*1000:.0f}ms\n20ms仿真步长 ≠ 实机20ms')
        for name,button in self.variant_buttons.items():button.set(('● ' if name==s.variant else '○ ')+VARIANTS[name])
        self.draw_chart(s)
    def zoom_ccd(self,age=0):
        if self.latest is None:return
        frame=self.latest.history[0].frame;stats=statistics96(frame.raw_counts)
        window=tk.Toplevel(self.root);window.title(f'96四幅 · 当前注册CCD #{frame.frame_id} · 固定模拟帧');window.configure(bg=PANEL)
        CjkLabel(window,'注册600×600模拟CCD；八区覆盖全场。标注、显示增益均不进入策略。',wrap=950).pack(padx=10,pady=8)
        body=tk.Frame(window,bg=PANEL);body.pack(padx=10)
        photo=ImageTk.PhotoImage(Image.fromarray(render96(frame.raw_counts,self.preview_gain,560)),master=window)
        label=tk.Label(body,image=photo,bg=PANEL);label.photo=photo;label.pack(side='left')
        side=tk.Frame(body,bg=PANEL);side.pack(side='left',anchor='n',padx=10)
        for i,(rect,count,q) in enumerate(zip(REGIONS,stats['region_counts'],stats['features'])):
            y0,y1,x0,x1=rect;CjkLabel(side,f'R{i+1} Σ={int(count)} q={q:.6f}\nx[{x0},{x1}) y[{y0},{y1})',size=11,color=REGION_COLORS[i],wrap=330).pack(anchor='w',pady=5)
        CjkLabel(window,'非原始相机ROI；历史LUT不代表当前实机标定；采样检查未通过。',color=WARNING,wrap=950).pack(padx=10,pady=8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--output',type=Path);p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--seed',type=int,default=92010);p.add_argument('--range',type=float,default=.05)
    p.add_argument('--duration',type=float,default=60.);p.add_argument('--running',action='store_true')
    p.add_argument('--quit-after',type=float);p.add_argument('--scripted-smoke',action='store_true');args=p.parse_args()
    if not os.environ.get('DISPLAY'):raise RuntimeError('real desktop X11 session required for Tk preview')
    output=args.output or rt.ROOT/'outputs/dashboard_sessions'/datetime.now().strftime('four96_%Y%m%d_%H%M%S_%f')
    worker=worker_module.FourFrameWorker(args.checkpoint,output,args.device,args.seed,args.range,args.duration,
        start_paused=not args.running,backend='simulation',flat_checkpoint=None)
    app=App96(worker,args.quit_after)
    def ready():
        if app.closing or app.closed:return
        if app.latest is None:app.root.after(120,ready);return
        app.root.lift();app.root.focus_force()
        # Let X11 paint the raised window before grabbing the desktop pixels.
        app.root.after(400,record_ready)
    def record_ready():
        if app.closing or app.closed:return
        shot=app.screenshot()
        rt.json_save(worker.output/'desktop_ready.json',dict(screenshot=str(shot),hardware_enabled=False,
            frame_id=app.latest.frame_id,variant=app.latest.variant,checkpoint_sha256=app.latest.checkpoint_sha256))
        if args.scripted_smoke:
            app.request('reset',xy=[-.044870167614183866,-.0061923431789715905])
            app.root.after(200,lambda:app.request('run'))
            app.root.after(2000,lambda:app.request('kick',direction=[1.,0.],speed=.35))
            app.root.after(4000,lambda:app.request('pause'))
            app.root.after(4500,lambda:app.request('save'))
            app.root.after(5000,lambda:(app.toggle_phase_color(),app.toggle_panels()))
            app.root.after(5500,app.toggle_phase_view)
            app.root.after(6000,lambda:(app.toggle_panels(),app.toggle_phase_color()))
            app.root.after(6300,lambda:app.request('variant',variant='flat_same_head'))
            app.root.after(6800,lambda:app.request('variant',variant='learned'))
            app.root.after(7300,app.screenshot)
            app.root.after(8000,lambda:app.close('scripted_smoke_complete'))
    app.root.after(120,ready)
    try:app.run()
    finally:
        if worker.is_alive():worker.shutdown('main_finally');worker.join()
    print(output,flush=True)


if __name__=='__main__':main()
