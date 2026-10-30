"""Commit the verified focused native GTK location entry with ONE Return.

No blind keys: exact original PID/window, ordinary monitor, active focus,
editable focused location field, complete pathname readback and file hash
are verified immediately before Return. Exact new GUI load event is required.
"""
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from night_opt_20261020 import manual_io as original


class ManualIO(original.ManualIO):
    def load(self,path,phase=False):
        self.guard(black=not bool(self.expected));path=Path(path).resolve();sha=original.digest(path)
        array=np.load(path,allow_pickle=False)
        if array.shape!=(600,600) or array.dtype!=np.uint8:raise ValueError('exact 600-square uint8 required')
        if phase and not np.array_equal(array,self.bundle.phase):raise ValueError('phase differs from declared bundle')
        if not phase and not np.isin(array,(0,255)).all():raise ValueError('nonbinary DMD input')
        before=len((original.SESSION/'events.jsonl').read_text().splitlines())
        title='载入规范SLM灰度图' if phase else '载入DMD图'
        button,Atspi=self.find('push button','载入mask' if phase else '载入图像')
        original.press(button);time.sleep(.2)
        dialogs=original.real_dialog(original.PID,title,Atspi)
        if len(dialogs)!=1:raise RuntimeError('one owned native optical-file dialog required')
        tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
        windows=[original.window(re.search(r'0x[0-9a-fA-F]+',line)[0]) for line in tree.splitlines() if '"'+title+'"' in line]
        windows=[w for w in windows if w['pid']==original.PID and w['viewable'] and original.inside(w['rect'],original.ORDINARY)]
        if len(windows)!=1:raise RuntimeError('one original owned ordinary-monitor file chooser required')
        wid=int(windows[0]['wid'],16);keyboard=original.Input(wid)
        def focused_window():
            actual=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
            if int(re.search(r'0x[0-9a-fA-F]+',actual)[0],16)!=wid:raise RuntimeError('native chooser lost actual keyboard focus')
        try:
            focused_window();keyboard.chord(ord('l'));time.sleep(.2)
            dialogs=original.real_dialog(original.PID,title,Atspi)
            if len(dialogs)!=1:raise RuntimeError('native dialog disappeared before filename edit')
            fields=[n for n in original.nodes(dialogs[0]) if n.get_role_name()=='text' and all(n.get_state_set().contains(s)
                for s in (Atspi.StateType.EDITABLE,Atspi.StateType.SHOWING,Atspi.StateType.FOCUSED))]
            if len(fields)!=1:raise RuntimeError('one focused native location field required')
            field=fields[0]
            if not field.get_editable_text_iface().set_text_contents(str(path)):raise RuntimeError('location edit rejected')
            time.sleep(.3)
            focused_window()
            if field.get_text_iface().get_text(0,-1)!=str(path) or not field.get_state_set().contains(Atspi.StateType.FOCUSED):
                raise RuntimeError('complete filename/focus readback failed')
            if original.digest(path)!=sha:raise RuntimeError('input file changed before activation')
            keyboard.key(0xff0d,True);keyboard.key(0xff0d,False)
        finally:keyboard.close()
        until=time.monotonic()+4
        while True:
            original.check_authority();original.alive()
            lines=(original.SESSION/'events.jsonl').read_text().splitlines()[before:];events=[]
            for i,line in enumerate(lines):
                try:events.append(json.loads(line))
                except json.JSONDecodeError:
                    if i!=len(lines)-1:raise
            if any(e.get('state')=='manual_file_loaded' and e.get('path')==str(path) and e.get('sha256')==sha and e.get('SLM')==phase for e in events):break
            if time.monotonic()>=until:raise TimeoutError('no exact GUI file-load acknowledgment after single focused Return')
            time.sleep(.05)
        if original.digest(path)!=sha:raise RuntimeError('loaded file changed')
        if phase:self.phase_loaded=True
        else:self.dmd=array.copy()
        self.expected=None
        return dict(path=str(path),sha256=sha,phase=phase,verified_native_location_Return_once=True,
            exact_new_GUI_load_event_acknowledged=True)
