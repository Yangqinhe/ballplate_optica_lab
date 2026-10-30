"""Visible v19 controls only: guarded native menus/buttons, no SDK/injection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from trigger_ui_v19 import PROJECT, inspect, nodes, ORDINARY
from select_verified_mode_v17 import window, inside, locate, extents, owned_popups, target_point, Pointer

NAMES=dict(ccd='仅CCD（硬触发）',arm='允许投图（ND／低功率条件已确认）',
           allzero='两路全0，CCD保持（Esc）',release='停采并恢复原参数',snapshot='保存当前对照',
           load_dmd='载入图像',load_slm='载入mask')
OPTIONS={('dmd','fiducial'):'非对称标定图',('dmd','file'):'载入的二值图',('dmd','black'):'全黑',
         ('slm','zero'):'固定灰度0',('slm','file'):'载入灰度图',('slm','+x'):'光栅 +X',
         ('slm','-x'):'光栅 −X',('slm','+y'):'光栅 +Y',('slm','-y'):'光栅 −Y'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OpticalUI:
    def __init__(self,session,before=None):
        self.session=Path(session)
        self.before=before or inspect(self.session,parked=True)
        self.pid=self.before['pid'];self.main=self.before['windows']['control']
        self.last_check=-float('inf')
        self.runtime_sha=digest(self.session/'runtime_sources.json')
        self.state()

    def state(self,allow_fault=False):
        h=json.loads((self.session/'heartbeat.json').read_text())
        if h['pid']!=self.pid or not 0<=time.time_ns()-h['unix_ns']<=5_000_000_000:
            raise RuntimeError('original v19 heartbeat/owner changed')
        if h['hardware_fault'] and not allow_fault:raise RuntimeError(h['hardware_fault'])
        if time.monotonic()-self.last_check>=1:
            if digest(self.session/'runtime_sources.json')!=self.runtime_sha:
                raise RuntimeError('runtime inventory changed')
            for group in ('sources','source_configs'):
                for path,sha in self.before['runtime'][group].items():
                    if digest(path)!=sha:raise RuntimeError('loaded source/config changed: '+path)
            for row in self.before['windows'].values():
                current=window(row['wid'])
                if current['pid']!=self.pid or not current['viewable'] or current['rect']!=row['rect']:
                    raise RuntimeError('original window moved/disappeared')
            self.last_check=time.monotonic()
        return h

    def fresh(self):
        h=self.state()
        return bool(h['live_camera'] and h.get('latest_frame') and h['latest_frame']['age_seconds']<1)

    def wait(self,predicate,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            h=self.state()
            if predicate(h):return h
            time.sleep(.1)
        raise RuntimeError('bounded visible UI transition timeout')

    def button(self,key,execute=True):
        h=self.state(allow_fault=key in ('allzero','release'))
        if key in ('ccd','load_dmd','load_slm') and (h['busy'] or h['live_camera'] or h['projection_armed']):
            raise RuntimeError('startup/file loading requires stopped disarmed owner')
        if key in ('arm','snapshot') and not self.fresh():raise RuntimeError('fresh CCD required')
        import gi
        gi.require_version('Atspi','2.0');from gi.repository import Atspi
        d=Atspi.get_desktop(0)
        apps=[d.get_child_at_index(i) for i in range(d.get_child_count())]
        apps=[a for a in apps if a.get_process_id()==self.pid and a.get_name()=='ballplate-realtime-manual']
        frames=[n for app in apps for n in nodes(app) if n.get_role_name()=='frame'
                and n.get_name().startswith('实时光路对齐 v19 ')]
        if len(frames)!=1:raise RuntimeError('unique PID-owned v19 main required')
        role='check box' if key=='arm' else 'push button'
        buttons=[n for n in nodes(frames[0]) if n.get_role_name()==role and n.get_name()==NAMES[key]]
        if len(buttons)!=1:raise RuntimeError('exact named control missing')
        node=buttons[0]
        for flag in (Atspi.StateType.ENABLED,Atspi.StateType.SENSITIVE,Atspi.StateType.SHOWING):
            if not node.get_state_set().contains(flag):raise RuntimeError('button disabled/hidden')
        if not inside(extents(node,Atspi),self.main['rect']):raise RuntimeError('button not in original main')
        iface=node.get_action_iface()
        choices=[i for i in range(iface.get_n_actions()) if iface.get_action_name(i) in ('Press','Toggle')]
        if len(choices)!=1 or (execute and not iface.do_action(choices[0])):raise RuntimeError('button action rejected')
        return dict(pid=self.pid,key=key,executed=execute,no_SDK=True,no_new_windows=True)

    def arm(self,value):
        if self.state()['projection_armed']!=value:self.button('arm')
        self.wait(lambda h:h['projection_armed']==value)

    def mode(self,role,mode,execute=True):
        h=self.state()
        if h['projection_armed'] or (h['busy'] and not h['live_camera']):
            raise RuntimeError('mode selection requires disarmed owner, no transition')
        if h['live_camera'] and not self.fresh():raise RuntimeError('stale CCD')
        if h['manual_settings'][role]==mode:
            return dict(pid=self.pid,role=role,mode=mode,no_change=True)
        combo,Atspi=locate(self.pid,role);rect=extents(combo,Atspi)
        if not inside(rect,self.main['rect']):raise RuntimeError('combo outside original main')
        name=OPTIONS[(role,mode)]
        result=dict(pid=self.pid,role=role,mode=mode,executed=execute)
        if not execute:return result
        for flag in (Atspi.StateType.ENABLED,Atspi.StateType.SENSITIVE,Atspi.StateType.SHOWING):
            if not combo.get_state_set().contains(flag):raise RuntimeError('combo disabled/hidden')
        if not owned_popups(self.pid,self.main['wid']):
            from calibration_desktop import Input
            focus=Input(int(self.main['wid'],16));focus.close()
            iface=combo.get_action_iface()
            indices=[i for i in range(iface.get_n_actions()) if iface.get_action_name(i)=='ShowMenu']
            if len(indices)!=1 or not iface.do_action(indices[0]):raise RuntimeError('native ShowMenu rejected')
        end=time.monotonic()+3;popups=[]
        while time.monotonic()<end:
            self.state();popups=owned_popups(self.pid,self.main['wid'])
            if popups:break
            time.sleep(.1)
        if len(popups)!=1:raise RuntimeError('unique native owned popup not found')
        popup=popups[0]
        if not (popup['rect'][0]<=rect[0]+rect[2]//2<popup['rect'][0]+popup['rect'][2]
                and popup['rect'][1]<rect[1]+rect[3] and rect[1]<popup['rect'][1]+popup['rect'][3]):
            raise RuntimeError('native popup not attached to requested combo')
        combo,Atspi=locate(self.pid,role)
        items=[n for n in nodes(combo) if n.get_role_name()=='list item' and n.get_name()==name]
        if len(items)!=1:raise RuntimeError('unique actual popup option not found')
        point=target_point(extents(items[0],Atspi),popup['rect'])
        self.state();pointer=Pointer()
        try:pointer.click(point,popup)
        finally:pointer.close()
        self.wait(lambda h:h['manual_settings'][role]==mode and not owned_popups(self.pid,self.main['wid']),5)
        return dict(result,popup=popup,native_popup_verified=True,heartbeat_committed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--execute',action='store_true')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('action');a.add_argument('key',choices=NAMES)
    m=sub.add_parser('mode');m.add_argument('role',choices=('dmd','slm'));m.add_argument('mode')
    args=p.parse_args();ui=OpticalUI(args.session)
    result=ui.button(args.key,args.execute) if args.cmd=='action' else ui.mode(args.role,args.mode,args.execute)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
