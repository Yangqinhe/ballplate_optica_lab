"""Desktop controls for one shared SLM and four RAW DMD times. Simulation only."""
import fourframe_runtime as rt
import argparse
import colorsys
from datetime import datetime
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageTk
from dashboard.app import DashboardApp
from dashboard.session import full_panel
from dashboard.widgets import (CjkButton,CjkLabel,ImagePanel,ACCENT,BACKGROUND,PANEL,MUTED,WARNING,FONT_PATH)
from readout_overlay import render_ccd,readout_statistics,REGION_COLORS
from fourframe_dashboard_session import DEFAULT_CHECKPOINT,DEFAULT_FLAT,VARIANTS
from fourframe_dashboard_worker import FourFrameWorker


PHASE_COLORS=np.asarray([np.array(colorsys.hsv_to_rgb(i/255,.85,.95))*255 for i in range(256)],dtype=np.uint8)


class FourFrameApp(DashboardApp):
    def __init__(self,*args,**kwargs):
        self.show_readout_regions=True;self.phase_color=True
        super().__init__(*args,**kwargs)
        self.preview_gain=8.
        self.root.title("四帧同屏 · 单SLM光学策略 · 手动抗扰动 · 仅仿真")

    def build(self):
        self.duration_var=tk.DoubleVar(value=.10);self.kick_var=tk.DoubleVar(value=.35)
        outer=ttk.Frame(self.root,padding=10);outer.pack(fill="both",expand=True)
        CjkLabel(outer,"四帧同屏 · 单SLM光学策略 · 手动抗扰动",size=23,background=BACKGROUND).pack(anchor="w")
        CjkLabel(outer,"4张原始图 → DMD 2×2拼图 → 单SLM一次传播 → 当前CCD 8区积分 → Linear(8,2) → 两轴控制",
                 size=13,color=MUTED,background=BACKGROUND).pack(anchor="w",pady=3)
        self.banner=CjkLabel(outer,"正在加载冻结模型；全仿真，无硬件SDK。",color=WARNING,background=BACKGROUND,wrap=1360)
        self.banner.pack(anchor="w",pady=(2,7))
        body=ttk.Frame(outer);body.pack(fill="both",expand=True)
        side=ttk.Frame(body,style="Panel.TFrame");side.pack(side="right",fill="y",padx=(10,0))
        side_canvas=tk.Canvas(side,width=300,bg=PANEL,highlightthickness=0)
        side_scroll=ttk.Scrollbar(side,orient="vertical",command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set);side_scroll.pack(side="right",fill="y")
        side_canvas.pack(side="left",fill="y",expand=True)
        sidebar=ttk.Frame(side_canvas,style="Panel.TFrame",padding=9)
        sid=side_canvas.create_window((0,0),window=sidebar,anchor="nw")
        sidebar.bind("<Configure>",lambda _:side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>",lambda e:side_canvas.itemconfigure(sid,width=e.width))
        left=ttk.Frame(body);left.pack(side="left",fill="both",expand=True)
        canvas=tk.Canvas(left,bg=BACKGROUND,highlightthickness=0)
        scroll=ttk.Scrollbar(left,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set);scroll.pack(side="right",fill="y");canvas.pack(fill="both",expand=True)
        content=ttk.Frame(canvas);cid=canvas.create_window((0,0),window=content,anchor="nw")
        content.bind("<Configure>",lambda _:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(cid,width=e.width))
        def wheel(e):
            target=side_canvas if self.root.winfo_pointerx()>=side.winfo_rootx() else canvas
            target.yview_scroll(-1 if e.delta>0 else 1,"units")
        canvas.bind_all("<MouseWheel>",wheel)
        for target in (canvas,side_canvas):
            target.bind("<Button-4>",lambda e,c=target:c.yview_scroll(-1,"units"))
            target.bind("<Button-5>",lambda e,c=target:c.yview_scroll(1,"units"))
        top=ttk.Frame(content);top.pack(fill="x")
        self.rgb=ImagePanel(top,"01 MuJoCo 环境",size=224,caption="诊断视角，不输入策略；策略使用顶视128²二值图")
        self.dmd=ImagePanel(top,"02 DMD 四时刻同屏",size=224,caption="逻辑256² / 有效512²；2×2微镜/逻辑点")
        self.slm=ImagePanel(top,"03 单SLM相位",size=224,caption="伪彩仅用于观察，未标定、不可部署")
        for i,panel in enumerate((self.rgb,self.dmd,self.slm)):
            top.columnconfigure(i,weight=1);panel.grid(row=0,column=i,sticky="nsew",padx=3)
        self.sync=CjkLabel(content,"等待当前观测",size=12,color=ACCENT,background=BACKGROUND,wrap=800)
        self.sync.pack(anchor="w",pady=(8,5))
        middle=ttk.Frame(content);middle.pack(fill="x")
        self.ccd_panel=ImagePanel(middle,"04 当前 CCD · R1–R8",size=320,caption="只显示当前帧；点击放大。没有CCD历史缓存。",accent=True)
        self.ccd_panel.pack(side="left",anchor="n",padx=3)
        self.ccd_panel.image_label.bind("<Button-1>",lambda _:self.zoom_ccd(0));self.ccd=[self.ccd_panel]
        sums=ttk.Frame(middle,style="Panel.TFrame",padding=10);sums.pack(side="left",anchor="n",fill="both",expand=True,padx=5)
        CjkLabel(sums,"05 当前8维光学读出",size=15,color=ACCENT).pack(anchor="w")
        self.readout_label=CjkLabel(sums,"等待当前CCD",size=11,color=MUTED,wrap=350);self.readout_label.pack(anchor="w",pady=5)
        readout_body=ttk.Frame(sums,style="Panel.TFrame");readout_body.pack(fill="both",expand=True)
        readout_values=ttk.Frame(readout_body,style="Panel.TFrame");readout_values.pack(side="left",anchor="n")
        self.monitor_column=ttk.Frame(readout_body,style="Panel.TFrame",padding=(12,0,0,0))
        self.monitor_column.pack(side="left",anchor="n",fill="both",expand=True)
        self.flux_chart=tk.Canvas(readout_values,width=315,height=230,bg="#0b1625",highlightthickness=0);self.flux_chart.pack()
        self.readout_note=CjkLabel(readout_values,"固定参考归一化，无隐藏MLP；8区覆盖整幅CCD。",size=11,color=MUTED,wrap=310)
        self.readout_note.pack(anchor="w",pady=6)
        view=ttk.Frame(content);view.pack(fill="x",pady=7)
        for label,callback in (("DMD整幅/有效ROI",self.toggle_panels),("SLM灰度/伪彩 M",self.toggle_phase_color),
                               ("CCD预览增益",self.toggle_gain),("显示/隐藏读出框 O",self.toggle_regions)):
            CjkButton(view,label,command=callback).pack(side="left",padx=3)
        CjkLabel(content,"四帧年龄为 60 / 40 / 20 / 0 ms；仅原始输入图保留历史。CCD只保留当前一张，不参与时序堆叠。",
                 size=12,color=MUTED,background=BACKGROUND,wrap=820).pack(anchor="w",pady=5)

        CjkLabel(sidebar,"运行与手动扰动",size=18,color=ACCENT).pack(anchor="w")
        self.state_label=CjkLabel(sidebar,"加载模型中…",wrap=280);self.state_label.pack(anchor="w",pady=4)
        self.pause_button=CjkButton(sidebar,"开始（空格）",command=self.toggle_run);self.pause_button.pack(fill="x",pady=2)
        self.step_button=CjkButton(sidebar,"单步20ms（Enter）",command=lambda:self.request("step"));self.step_button.pack(fill="x",pady=2)
        resetrow=ttk.Frame(sidebar,style="Panel.TFrame");resetrow.pack(fill="x",pady=2)
        self.center_button=CjkButton(resetrow,"中心复位 R",command=lambda:self.request("reset",random=False));self.center_button.pack(side="left",fill="x",expand=True)
        self.random_button=CjkButton(resetrow,"随机复位 N",command=lambda:self.request("reset",random=True));self.random_button.pack(side="left",fill="x",expand=True,padx=(3,0))
        presets=ttk.Frame(sidebar,style="Panel.TFrame");presets.pack(fill="x",pady=2)
        self.far_button=CjkButton(presets,"+12cm轴向复位",command=lambda:self.request("reset",xy=[.12,0.]));self.far_button.pack(side="left",fill="x",expand=True)
        self.corner_button=CjkButton(presets,"四角复位",command=lambda:self.request("reset",xy=[.12,-.12]));self.corner_button.pack(side="left",fill="x",expand=True,padx=(3,0))
        CjkLabel(sidebar,"力脉冲：板面XY（WASD/方向键）",size=13,color=ACCENT).pack(anchor="w",pady=(9,2))
        self.force_label=CjkLabel(sidebar,"",size=11,color=MUTED,wrap=280);self.force_label.pack(anchor="w")
        ttk.Scale(sidebar,from_=.01,to=.25,variable=self.force_var,command=self.force_changed).pack(fill="x")
        ttk.Scale(sidebar,from_=.02,to=.5,variable=self.duration_var,command=self.force_changed).pack(fill="x")
        arrows=ttk.Frame(sidebar,style="Panel.TFrame");arrows.pack(fill="x",pady=2);self.push_buttons=[]
        for i,(label,direction) in enumerate((("−X",[-1,0]),("+Y",[0,1]),("−Y",[0,-1]),("+X",[1,0]))):
            button=CjkButton(arrows,label,command=lambda d=direction:self.push(d));button.grid(row=0,column=i,padx=1);self.push_buttons.append(button)
        CjkLabel(sidebar,"瞬时速度扰动：世界XY",size=13,color=ACCENT).pack(anchor="w",pady=(7,2))
        self.kick_label=CjkLabel(sidebar,"",size=11,color=MUTED);self.kick_label.pack(anchor="w")
        ttk.Scale(sidebar,from_=.01,to=.5,variable=self.kick_var,command=self.kick_changed).pack(fill="x")
        kicks=ttk.Frame(sidebar,style="Panel.TFrame");kicks.pack(fill="x",pady=2);self.kick_buttons=[]
        for i,(label,direction) in enumerate((("−X",[-1,0]),("+Y",[0,1]),("−Y",[0,-1]),("+X",[1,0]))):
            button=CjkButton(kicks,label,command=lambda d=direction:self.kick(d));button.grid(row=0,column=i,padx=1);self.kick_buttons.append(button)
        self.release_button=CjkButton(sidebar,"释放/取消持续外力 F",command=lambda:self.request("release"));self.release_button.pack(fill="x",pady=3)
        CjkLabel(sidebar,"相位消融切换（复位并暂停）",size=13,color=ACCENT).pack(anchor="w",pady=(8,2))
        self.variant_buttons={}
        for name,label in VARIANTS.items():
            button=CjkButton(sidebar,label,command=lambda n=name:self.request("variant",variant=n));button.pack(fill="x",pady=2);self.variant_buttons[name]=button
        self.control_label=CjkLabel(self.monitor_column,"等待控制输出",wrap=260,size=12);self.control_label.pack(anchor="w",pady=(0,3))
        self.chart=tk.Canvas(self.monitor_column,width=254,height=76,bg="#0b1625",highlightthickness=0);self.chart.pack()
        CjkLabel(self.monitor_column,"蓝/橙：roll/pitch；实线下一输出，虚线上次执行",size=10,color=MUTED,wrap=260).pack(anchor="w")
        self.diagnostic_label=CjkLabel(self.monitor_column,"诊断位置/速度不输入策略",size=11,color=MUTED,wrap=260);self.diagnostic_label.pack(anchor="w",pady=5)
        self.performance_label=CjkLabel(self.monitor_column,"仿真20ms ≠ 墙钟20ms",size=11,color=WARNING,wrap=260);self.performance_label.pack(anchor="w",pady=3)
        self.save_button=CjkButton(sidebar,"保存当前帧与日志 P",command=self.save);self.save_button.pack(fill="x",pady=3)
        self.notice=CjkLabel(sidebar,"全仿真；复位后暂停。",size=11,color=MUTED,wrap=280);self.notice.pack(anchor="w",pady=5)
        self.action_buttons=[self.pause_button,self.step_button,self.center_button,self.random_button,
            self.far_button,self.corner_button,self.release_button,self.save_button,*self.push_buttons,*self.kick_buttons,*self.variant_buttons.values()]
        for button in self.action_buttons:button.configure(state="disabled")
        self.force_changed();self.kick_changed()

    def force_changed(self,*_):
        f,d=float(self.force_var.get()),float(self.duration_var.get())
        self.force_label.set(f"上滑块：{f:.3f} N；下滑块：{d*1000:.0f} ms\n冲量 {f*d:.4f} N·s（不等同实际Δv）")

    def kick_changed(self,*_):self.kick_label.set(f"Δv = {self.kick_var.get():.3f} m/s（默认0.35）")

    def push(self,direction):
        self.request("push",direction=direction,force=float(self.force_var.get()),duration=float(self.duration_var.get()))
        self.notice.set("力脉冲按仿真时间推进；暂停时等待开始/单步。")

    def kick(self,direction):
        self.request("kick",direction=direction,speed=float(self.kick_var.get()))
        self.notice.set("已请求瞬时Δv；暂停时按开始/单步观察。释放F不会撤销已经注入的速度。")

    def key(self,event):
        if event.keysym.lower()=="o":self.toggle_regions();return "break"
        if event.keysym.lower()=="m":self.toggle_phase_color();return "break"
        return super().key(event)

    def toggle_phase_color(self):
        self.phase_color=not self.phase_color
        if self.latest:self.draw(self.latest)

    def toggle_regions(self):
        self.show_readout_regions=not self.show_readout_regions
        if self.latest:self.draw(self.latest)

    def toggle_gain(self):
        self.preview_gain={1.:4.,4.:8.,8.:16.,16.:1.}[self.preview_gain]
        if self.latest:self.draw(self.latest)

    def draw(self,s):
        self.paint_count+=1;self.rgb.set_array(s.rgb)
        if self.native_panels:
            self.dmd.set_array(full_panel(s.dmd_active,0),outline=(704,284,512,512))
        else:
            dmd=Image.fromarray(s.dmd_active).convert("RGB");draw=ImageDraw.Draw(dmd)
            font=ImageFont.truetype(str(FONT_PATH),24)
            for i,(age,color) in enumerate(zip((60,40,20,0),REGION_COLORS[:4])):
                x,y=(i%2)*256,(i//2)*256
                draw.rectangle((x,y,x+255,y+255),outline=color,width=3)
                draw.text((x+7,y+6),f"{'当前' if age==0 else f'-{age}ms'} #{s.raw_input_ids[i]}",font=font,fill=color)
            self.dmd.set_array(np.array(dmd))
        fill="冷启动重复填充；" if len(set(s.raw_input_ids))<4 else ""
        self.dmd.caption.set(f"256²逻辑 / 512²微镜 / 2×2每点\n{fill}输入ID {list(s.raw_input_ids)}；标注不进入光路")
        self.slm.set_array(PHASE_COLORS[s.slm_active] if self.phase_color else s.slm_active)
        self.slm.caption.set(f"{'相位伪彩（仅显示）' if self.phase_color else '原始灰度（仅预览）'}；灰度{s.phase_gray_range[0]}–{s.phase_gray_range[1]} / {s.phase_levels}级\n512²有效区；真实LUT未标定")
        mode=VARIANTS[s.variant]
        numeric=self.metadata.get("known_main_model_sampling_action_difference_rad")
        warning="采样精度未验收" if numeric is None else f"主候选采样差{numeric*1000:.0f}mrad，未通过2mrad门限"
        self.banner.set(f"全仿真 · {mode} · 非实机/非部署结果：{warning}")
        ended={"missing_optical_signal":"缺少光学信号，已停止","fell":"越界/掉球终止，请复位",
               "episode_complete":"演示回合完成，请复位"}
        status=ended.get(s.end_reason,"回合结束，请复位") if s.finished else "闭环运行" if self.running else "已暂停"
        self.state_label.set(f"{status} · {mode}\n回合{s.episode} / t={s.simulated_time_s:.2f}s\n当前观测 #{s.frame_id}；仅推理，不训练")
        self.pause_button.set("请复位" if s.finished else "暂停（空格）" if self.running else "开始（空格）")
        for button in [self.pause_button,self.step_button,*self.push_buttons,*self.kick_buttons]:button.configure(state="disabled" if s.finished else "normal")
        regions=self.metadata["ccd_readout"]["corrected_regions"]
        frame=s.history[0].frame;stats=readout_statistics(frame.raw_counts,regions,self.metadata["ccd_readout"]["reference_counts"])
        if not np.array_equal(stats["features"],frame.features):raise RuntimeError("display readout differs from actual actor input")
        self.ccd_panel.set_array(render_ccd(frame.raw_counts,regions,gain=self.preview_gain,size=320,
            show_regions=self.show_readout_regions,labels=True))
        self.ccd_panel.caption.set(f"原生CCD 1280² Mono8 / 增益{self.preview_gain:g}×仅显示\n#{s.frame_id} t={s.simulated_time_s:.3f}s；点击放大")
        self.readout_label.set(f"q = 当前8区计数 ÷ {self.metadata['ccd_readout']['reference_counts']:g}\n显示/策略一致误差 {s.cache_error:.1g}；无CCD历史")
        q=frame.features;self.flux_chart.delete("all");limit=max(float(q.max()),1e-8)
        for i,(count,value) in enumerate(zip(stats["region_counts"],q)):
            y=10+i*27;color=REGION_COLORS[i]
            self.flux_chart.create_rectangle(30,y+2,30+max(1,float(value)/limit*108),y+15,fill=color,outline="")
            self.flux_chart.create_text(2,y+8,anchor="w",text=f"R{i+1}",fill=color)
            self.flux_chart.create_text(148,y+8,anchor="w",text=f"q={value:.5f}  Σ={int(count)}",fill="#e7eff8",font=("TkFixedFont",10))
        self.readout_note.set("8区覆盖整幅CCD：1,638,400像素求和。\n只减少FC输入，不等于相机只读8个像素。\n共享SLM，区域不等于独立时间通道。")
        self.sync.set(f"当前CCD8 → 18参数FC；仅输入原图缓存4帧（60/40/20/0ms）｜显示CCD仅1帧｜前向累计{s.optical_forward_calls}次")
        pending,applied=s.next_mean_action,s.last_applied_action
        self.control_label.set(f"当前CCD → 下一策略均值(rad)\nroll {pending[0]:+.5f}　pitch {pending[1]:+.5f}\n上一周期执行（低通/限幅后）\nroll {applied[0]:+.5f}　pitch {applied[1]:+.5f}")
        self.diagnostic_label.set(f"诊断 x={s.position[0]*100:+.2f} y={s.position[1]*100:+.2f} cm\n速度{s.speed*100:.2f}cm/s；稳定{s.stable}\n奖励{s.reward_sum:.1f}；外力剩{s.force_steps_remaining}步\n真实位置/速度不输入策略")
        self.performance_label.set(f"{s.propagation} {s.grid_size} / {s.device}\n本帧光学{s.capture_seconds*1000:.0f}ms；均步{s.mean_step_seconds*1000:.0f}ms\n20ms仿真步长 ≠ 墙钟20ms/硬件延迟")
        for name,button in self.variant_buttons.items():
            button.set(("● " if name==s.variant else "○ ")+VARIANTS[name])
            if name not in self.metadata["available_variants"]:button.configure(state="disabled")
        self.draw_chart(s)

    def zoom_ccd(self,age=0):
        if self.latest is None:return
        frame=self.latest.history[0].frame;regions=self.metadata["ccd_readout"]["corrected_regions"]
        stats=readout_statistics(frame.raw_counts,regions,self.metadata["ccd_readout"]["reference_counts"])
        window=tk.Toplevel(self.root);window.title(f"四帧光学策略 · 当前CCD #{frame.frame_id} · 固定帧")
        window.configure(bg=PANEL)
        CjkLabel(window,"模拟CCD固定帧；标注/亮度只用于显示，不进入策略。单共享SLM，不是四个独立卷积。",wrap=1000).pack(padx=10,pady=8)
        body=tk.Frame(window,bg=PANEL);body.pack(padx=10)
        label=tk.Label(body,bg=PANEL);label.pack(side="left")
        side=tk.Frame(body,bg=PANEL);side.pack(side="left",anchor="n",padx=10)
        corrected=[False]
        def paint():
            array=render_ccd(frame.raw_counts,regions,gain=self.preview_gain,size=560,show_regions=True,corrected_view=corrected[0])
            label.photo=ImageTk.PhotoImage(Image.fromarray(array),master=window);label.configure(image=label.photo)
        def rotate():corrected[0]=not corrected[0];paint()
        CjkButton(side,"原生 / 180°校正朝向",command=rotate).pack(fill="x",pady=4)
        for i,(rect,count,value) in enumerate(zip(regions,stats["region_counts"],stats["features"])):
            y0,y1,x0,x1=rect
            CjkLabel(side,f"R{i+1}  Σ={int(count)}  q={value:.6f}\n校正坐标 x[{x0},{x1}) y[{y0},{y1})",size=11,color=REGION_COLORS[i],wrap=360).pack(anchor="w",pady=5)
        CjkLabel(window,"仅当前8维q进入线性头；没有CCD历史。真实LUT/配准未测，采样复核失败，禁止据此直接部署。",wrap=1000,size=11,color=WARNING).pack(padx=10,pady=8)
        paint()


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument("--flat-checkpoint",type=Path,default=DEFAULT_FLAT);p.add_argument("--output",type=Path)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--seed",type=int,default=92010)
    p.add_argument("--range",type=float,default=.05);p.add_argument("--duration",type=float,default=60.)
    p.add_argument("--running",action="store_true");p.add_argument("--quit-after",type=float)
    args=p.parse_args()
    if not os.environ.get("DISPLAY"):raise RuntimeError("X11 desktop is required")
    output=args.output or rt.ROOT/"outputs/dashboard_sessions"/datetime.now().strftime("fourframe_%Y%m%d_%H%M%S_%f")
    worker=FourFrameWorker(args.checkpoint,output,args.device,args.seed,args.range,args.duration,
        start_paused=not args.running,backend="simulation",flat_checkpoint=args.flat_checkpoint)
    app=FourFrameApp(worker,args.quit_after)
    def capture_ready():
        if app.closing or app.closed:return
        if app.latest is not None:
            app.root.lift();app.root.focus_force()
            shot=app.screenshot()
            rt.json_save(worker.output/"desktop_ready.json",dict(screenshot=str(shot),
                frame_id=app.latest.frame_id,simulation_time_s=app.latest.simulated_time_s,
                running=app.running,variant=app.latest.variant,hardware_enabled=False,
                actor_parameters=18,ccd_history_used_by_actor=0,ccd_frames_retained=1))
        else:app.root.after(120,capture_ready)
    app.root.after(120,capture_ready)
    try:app.run()
    finally:
        if worker.is_alive():worker.shutdown("main_finally");worker.join()
    print(str(output),flush=True)


if __name__=="__main__":main()
