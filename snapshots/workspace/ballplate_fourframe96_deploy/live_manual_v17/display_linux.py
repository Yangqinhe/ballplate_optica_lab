"""Read-only XRandR identity/timing backend for explicit raw diagnostics.

Not a fake Windows video fingerprint and not a calibration release token.
"""
import hashlib
import re
import subprocess
import numpy as np

ROLES={'dmd':('DP-1-2','DLP6500 DP',(800,600)),
       'slm':('DP-1-1','XILINX HDMI',(1920,1080))}


def parse_xrandr(text):
    result={}
    for block in re.split(r'(?m)(?=^\S+ (?:connected|disconnected))',text):
        lines=block.splitlines()
        if not lines:continue
        m=re.match(r'^(\S+) connected (?:primary )?(\d+)x(\d+)\+(-?\d+)\+(-?\d+) \([^)]*\) (\S+)',lines[0])
        if not m:continue
        name=m[1];width,height,x,y=map(int,m.group(2,3,4,5))
        if name not in {role[0] for role in ROLES.values()}:continue
        edid=re.search(r'\n\s+EDID:\s*\n((?:\s+[0-9a-fA-F]{32}\s*\n)+)',block)
        if not edid:raise RuntimeError(name+': missing EDID')
        data=bytes.fromhex(edid[1]);target=''
        for pos in (54,72,90,108):
            descriptor=data[pos:pos+18]
            if len(descriptor)==18 and descriptor[:4]==b'\x00\x00\x00\xfc':
                target=descriptor[5:].decode('ascii').strip()
        mode=re.search(r'^\s+\d+x\d+ \([^)]*\) ([\d.]+)MHz[^\n]*\*current[^\n]*\n'
                       r'\s+h: width\s+(\d+).*?total\s+(\d+)[^\n]*\n'
                       r'\s+v: height\s+(\d+).*?total\s+(\d+)',block,re.M)
        transform=re.search(r'\n\s+Transform:\s+([^\n]+)\n\s+([^\n]+)\n\s+([^\n]+)',block)
        if not mode or not transform:raise RuntimeError(name+': timing/transform readback incomplete')
        matrix=np.array([[float(v) for v in transform[i].split()] for i in (1,2,3)])
        if matrix.shape!=(3,3) or not np.array_equal(matrix,np.eye(3)):raise RuntimeError(name+': nonidentity scaling transform')
        if m[6]!='normal':raise RuntimeError(name+': rotated output is not accepted')
        clock=round(float(mode[1])*1e6);aw,tw,ah,th=map(int,mode.group(2,3,4,5))
        if (aw,ah)!=(width,height):raise RuntimeError(name+': desktop/signal dimensions differ')
        result[name]=dict(connector=name,target_name=target,desktop_size=[width,height],
            desktop_position=[x,y],signal_active_size=[aw,ah],signal_total_size=[tw,th],
            pixel_clock_hz=clock,refresh_hz=clock/(tw*th),rotation='normal',transform=matrix.tolist(),
            EDID_sha256=hashlib.sha256(data).hexdigest(),framebuffer_is_not_photometric_calibration=True)
    return result


def read_roles():
    # --verbose alone polls hardware (XRRGetScreenResources). With these DP
    # devices/NVIDIA drivers that blocks the entire X server for ~1.4 seconds,
    # including the physical mouse, even if called from a background thread.
    # --current reads the live server configuration without hardware probing.
    # EDID, geometry, timings and transforms are still checked below.
    all_displays=parse_xrandr(subprocess.check_output(['xrandr','--current','--verbose'],text=True,timeout=3))
    selected={}
    for role,(connector,target,size) in ROLES.items():
        row=all_displays.get(connector)
        if row is None or row['target_name']!=target or tuple(row['desktop_size'])!=size:
            raise RuntimeError(f'{role}: connector/EDID-name/active-size mismatch')
        selected[role]=row
    rectangles=[]
    for row in selected.values():
        x,y=row['desktop_position'];w,h=row['desktop_size'];rectangles.append((x,y,x+w,y+h))
    a,b=rectangles
    if min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):
        raise RuntimeError('DMD and SLM outputs overlap/mirror')
    return selected


def same_signals(saved,current):
    # Unlike desktop layout, geometry in this diagnostic is safety-critical too.
    if saved!=current:raise RuntimeError('display identity, mode or layout changed: stop diagnostic')
