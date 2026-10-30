"""Read-only AT-SPI traversal that discards only known defunct objects.

No target-process injection, SDK, new windows, or repeated actions. Old GTK
file dialogs can remain in client-side accessibility caches after destruction.
Fresh child lists and exact target uniqueness are still mandatory.
"""
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from night_opt_20261020 import manual_io as original
from night_opt_20261020.manual_io_v13 import ManualIO as CausalIO, next_revision_floor
PID=original.PID
alive=original.alive
press=original.press


def defunct(error):
    return 'atspi_error' in str(error) and 'does not exist' in str(error)


def safe_nodes(root):
    pending=[root]
    while pending:
        node=pending.pop()
        try:
            node.clear_cache()
            node.get_role_name(); node.get_name()
            count=node.get_child_count()
            children=[node.get_child_at_index(i) for i in range(count)]
        except Exception as error:
            if defunct(error): continue
            raise
        yield node
        pending.extend(reversed([c for c in children if c is not None]))


def owned_apps(pid):
    import gi
    gi.require_version('Atspi','2.0');from gi.repository import Atspi
    desktop=Atspi.get_desktop(0);desktop.clear_cache();apps=[]
    for i in range(desktop.get_child_count()):
        try:
            app=desktop.get_child_at_index(i);app.clear_cache()
            if app.get_process_id()==pid:apps.append(app)
        except Exception as error:
            if not defunct(error):raise
    return apps,Atspi


def accessible(pid):
    apps,Atspi=owned_apps(pid);selected=[]
    for app in apps:
        try:
            if app.get_name()=='ballplate-realtime-manual':selected.append(app)
        except Exception as error:
            if not defunct(error):raise
    if len(selected)!=1:raise RuntimeError('one original accessible control owner required')
    return selected[0],Atspi


def real_dialog(pid,title,Atspi):
    apps,_=owned_apps(pid);dialogs=[]
    for app in apps:
        for node in safe_nodes(app):
            try:
                if node.get_role_name() in ('dialog','file chooser') and node.get_name()==title and node.get_child_count()>0:
                    dialogs.append(node)
            except Exception as error:
                if not defunct(error):raise
    if len(dialogs)>1:raise RuntimeError('ambiguous current native dialogs')
    return dialogs


def focused_fields(dialog,Atspi):
    fields=[]
    for n in safe_nodes(dialog):
        try:
            if n.get_role_name()=='text' and all(n.get_state_set().contains(s) for s in
                (Atspi.StateType.EDITABLE,Atspi.StateType.SHOWING,Atspi.StateType.FOCUSED)):fields.append(n)
        except Exception as error:
            if not defunct(error):raise
    return fields


class ManualIO(CausalIO):
    def find(self,role,name,showing=True):
        app,Atspi=accessible(PID);matches=[]
        for node in safe_nodes(app):
            try:
                if node.get_role_name()==role and node.get_name()==name:matches.append(node)
            except Exception as error:
                if not defunct(error):raise
        if len(matches)!=1:raise RuntimeError('missing/ambiguous current native control: '+name)
        node=matches[0]
        if showing:
            for flag in (Atspi.StateType.SHOWING,Atspi.StateType.SENSITIVE,Atspi.StateType.ENABLED):
                if not node.get_state_set().contains(flag):raise RuntimeError('control hidden/disabled: '+name)
            main=original.window(self.windows['control'])
            if not original.inside(original.extents(node,Atspi),main['rect']):raise RuntimeError('control outside original main')
        return node,Atspi
    def load(self,path,phase=False):
        heartbeat,_=self.guard(black=not bool(self.expected))
        floor=next_revision_floor(heartbeat['display_revision'],self.last_acknowledged_revision)
        path=Path(path).resolve();sha=original.digest(path)
        array=np.load(path,allow_pickle=False)
        if array.shape!=(600,600) or array.dtype!=np.uint8:raise ValueError('exact 600-square uint8 required')
        if phase and not np.array_equal(array,self.bundle.phase):raise ValueError('phase differs from declared bundle')
        if not phase and not np.isin(array,(0,255)).all():raise ValueError('nonbinary DMD input')
        before=len((original.SESSION/'events.jsonl').read_text().splitlines())
        title='载入规范SLM灰度图' if phase else '载入DMD图'
        button,Atspi=self.find('push button','载入mask' if phase else '载入图像')
        original.press(button);time.sleep(.2)
        dialogs=real_dialog(original.PID,title,Atspi)
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
            dialogs=real_dialog(original.PID,title,Atspi)
            if len(dialogs)!=1:raise RuntimeError('native dialog disappeared before filename edit')
            fields=focused_fields(dialogs[0],Atspi)
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
        self.expected=None;self.minimum_revision=floor
        return dict(path=str(path),sha256=sha,phase=phase,verified_native_location_Return_once=True,
            exact_new_GUI_load_event_acknowledged=True,required_new_display_revision_at_least=floor)
