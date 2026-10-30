"""External-client pointer guard: ignore ONLY a verified NumLock modifier.

Does not change keyboard locks or weaken target-window, coordinate, button,
Ctrl/Shift/Alt/Super guards. No code is injected into the optical GUI.
"""
import ctypes as C
import re
import subprocess
import time


def numlock_mask(mapping):
    lines=[line for line in mapping.splitlines() if line.startswith('mod2')]
    if len(lines)!=1:raise RuntimeError('one modifier2 mapping required')
    names=re.findall(r'([A-Za-z_][A-Za-z_0-9]*)\s+\(0x[0-9a-fA-F]+\)',lines[0])
    if not names or set(names)!={'Num_Lock'}:
        raise RuntimeError('mod2 contains keys other than verified NumLock')
    return 1<<4


def valid_modifiers(actual,allowed_numlock):
    return allowed_numlock==16 and actual>=0 and (actual & ~allowed_numlock)==0


def install():
    import optical_ui_v19 as client
    mapping=subprocess.check_output(['xmodmap','-pm'],text=True)
    allowed=numlock_mask(mapping)
    original=client.Pointer

    class NumLockPointer(original):
        def click(self,point,popup):
            x,y=point
            if not self.t.XTestFakeMotionEvent(self.d,-1,x,y,0):raise RuntimeError('pointer move rejected')
            self.x.XSync(self.d,False);time.sleep(.12)
            root=C.c_ulong();child=C.c_ulong();rx=C.c_int();ry=C.c_int();wx=C.c_int();wy=C.c_int();mask=C.c_uint()
            ok=self.x.XQueryPointer(self.d,self.root,C.byref(root),C.byref(child),C.byref(rx),C.byref(ry),C.byref(wx),C.byref(wy),C.byref(mask))
            if not ok or child.value!=int(popup['wid'],16) or (rx.value,ry.value)!=(x,y) or not valid_modifiers(mask.value,allowed):
                raise RuntimeError('actual target window/coordinates/active modifiers differ; no click')
            if not self.t.XTestFakeButtonEvent(self.d,1,True,0):raise RuntimeError('mouse press rejected')
            try:self.x.XSync(self.d,False);time.sleep(.08)
            finally:self.t.XTestFakeButtonEvent(self.d,1,False,0);self.x.XSync(self.d,False)

    client.Pointer=NumLockPointer
    return dict(external_client_only=True,ignored_modifier='verified NumLock ONLY',
                ignored_mask=allowed,keyboard_lock_state_unchanged=True,xmodmap=mapping)
