"""Read-only Windows display signal timing; does NOT set modes or sync devices.

Layouts follow Microsoft wingdi.h DISPLAYCONFIG structures. Report rational
vertical frequency instead of rounding both 59.94 and 60 Hz to '60'.
"""
import ctypes as c
from datetime import datetime,timezone

U=c.c_uint32
I=c.c_int32

class Luid(c.Structure):
    _fields_=[('low',U),('high',I)]
class Rational(c.Structure):
    _fields_=[('numerator',U),('denominator',U)]
class Region(c.Structure):
    _fields_=[('width',U),('height',U)]
class Point(c.Structure):
    _fields_=[('x',I),('y',I)]
class Signal(c.Structure):
    _fields_=[('pixel_rate',c.c_uint64),('h_sync',Rational),('v_sync',Rational),
              ('active',Region),('total',Region),('standard',U),('scan_order',U)]
class SourceMode(c.Structure):
    _fields_=[('width',U),('height',U),('pixel_format',U),('position',Point)]
class ModeUnion(c.Union):
    _fields_=[('target',Signal),('source',SourceMode)]
class Mode(c.Structure):
    _fields_=[('kind',U),('id',U),('adapter',Luid),('value',ModeUnion)]
class PathSource(c.Structure):
    _fields_=[('adapter',Luid),('id',U),('mode_index',U),('flags',U)]
class PathTarget(c.Structure):
    _fields_=[('adapter',Luid),('id',U),('mode_index',U),('technology',U),
              ('rotation',U),('scaling',U),('refresh',Rational),('scan_order',U),
              ('available',I),('flags',U)]
class DisplayPath(c.Structure):
    _fields_=[('source',PathSource),('target',PathTarget),('flags',U)]
class Header(c.Structure):
    _fields_=[('kind',U),('size',U),('adapter',Luid),('id',U)]
class SourceName(c.Structure):
    _fields_=[('header',Header),('name',c.c_wchar*32)]
class TargetName(c.Structure):
    _fields_=[('header',Header),('flags',U),('technology',U),
              ('manufacturer',c.c_uint16),('product',c.c_uint16),('connector',U),
              ('name',c.c_wchar*64),('device_path',c.c_wchar*128)]


def rational(value):
    return {'numerator':value.numerator,'denominator':value.denominator,
            'hz':value.numerator/value.denominator if value.denominator else None}


def read_display_timing(include_available_modes=False):
    if c.sizeof(c.c_wchar)!=2:raise RuntimeError('Windows only')
    for cls,expected in ((Signal,48),(Mode,64),(DisplayPath,72),(SourceName,84),(TargetName,420)):
        if c.sizeof(cls)!=expected:raise RuntimeError('Unexpected Windows layout: '+cls.__name__)
    user=c.WinDLL('user32',use_last_error=True)
    user.GetDisplayConfigBufferSizes.argtypes=[U,c.POINTER(U),c.POINTER(U)]
    user.QueryDisplayConfig.argtypes=[U,c.POINTER(U),c.POINTER(DisplayPath),c.POINTER(U),c.POINTER(Mode),c.c_void_p]
    user.DisplayConfigGetDeviceInfo.argtypes=[c.c_void_p]
    for method in (user.GetDisplayConfigBufferSizes,user.QueryDisplayConfig,user.DisplayConfigGetDeviceInfo):method.restype=I
    for _ in range(3):
        paths_count=U();modes_count=U()
        code=user.GetDisplayConfigBufferSizes(2,c.byref(paths_count),c.byref(modes_count))
        if code:raise OSError(code,'GetDisplayConfigBufferSizes')
        paths=(DisplayPath*paths_count.value)();modes=(Mode*modes_count.value)()
        code=user.QueryDisplayConfig(2,c.byref(paths_count),paths,c.byref(modes_count),modes,None)
        if code==122:continue
        if code:raise OSError(code,'QueryDisplayConfig')
        break
    else:raise RuntimeError('Display topology kept changing')
    rows=[]
    for path in paths[:paths_count.value]:
        source_name=SourceName(Header(1,c.sizeof(SourceName),path.source.adapter,path.source.id))
        target_name=TargetName(Header(2,c.sizeof(TargetName),path.target.adapter,path.target.id))
        source_code=user.DisplayConfigGetDeviceInfo(c.byref(source_name))
        target_code=user.DisplayConfigGetDeviceInfo(c.byref(target_name))
        row={'source_name':source_name.name,'target_name':target_name.name,
            'name_query_codes':[source_code,target_code],'path_refresh':rational(path.target.refresh),
            'scaling_enum':path.target.scaling,'rotation_enum':path.target.rotation}
        if path.source.mode_index<modes_count.value:
            mode=modes[path.source.mode_index]
            if mode.kind!=1:raise RuntimeError('Invalid source mode type')
            src=mode.value.source
            row['desktop_size']=[src.width,src.height];row['desktop_position']=[src.position.x,src.position.y]
        if path.target.mode_index<modes_count.value:
            mode=modes[path.target.mode_index]
            if mode.kind!=2:raise RuntimeError('Invalid target mode type')
            signal=mode.value.target
            row.update({'signal_refresh':rational(signal.v_sync),'pixel_clock_hz':signal.pixel_rate,
                'signal_active_size':[signal.active.width,signal.active.height],
                'signal_total_size':[signal.total.width,signal.total.height],
                'scan_order_enum':signal.scan_order})
        if include_available_modes:
            # EnumDisplaySettings is integer-Hz only; use QueryDisplayConfig above
            # for the active rational rate. Listing supported modes changes nothing.
            import win32api
            import pywintypes
            choices=set()
            for index in range(4096):
                try:mode=win32api.EnumDisplaySettings(source_name.name,index)
                except pywintypes.error:break
                if [mode.PelsWidth,mode.PelsHeight]==row.get('desktop_size'):
                    choices.add((mode.DisplayFrequency,mode.BitsPerPel,mode.DisplayFlags))
            row['same_resolution_advertised_modes']=[{'integer_hz':hz,'bits_per_pixel':bpp,'flags':flags}
                for hz,bpp,flags in sorted(choices)]
        rows.append(row)
    return {'time_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,
            'meaning':'OS video timings, not measured liquid-crystal or DMD trigger timing','displays':rows}


def display_mode_signature(state,role_sources):
    """Bind optical roles to signal modes, independent of desktop placement.

    Changing monitor numbering/desktop placement alone need not change the
    physical optical mapping. Active signal timings and target identity do.
    """
    rows={row['source_name'].lower():row for row in state['displays']}
    result={'format_version':1}
    keys=('target_name','desktop_size','signal_active_size','signal_total_size',
          'pixel_clock_hz','rotation_enum','scaling_enum')
    for role in ('dmd','slm'):
        row=rows.get(str(role_sources[role]).lower())
        if row is None:raise ValueError('无法读取 %s 视频输出；请检查屏幕连接与分配' % role.upper())
        try:
            value={key:row[key] for key in keys}
            rate=row['signal_refresh']
            value['refresh_hz']=round(rate['numerator']/rate['denominator'],6)
        except (KeyError,TypeError,ZeroDivisionError):
            raise ValueError('%s 视频信号参数不完整，不能验证标定' % role.upper()) from None
        if value['refresh_hz']<=0:raise ValueError('无效的视频刷新率')
        result[role]=value
    return result


def assert_display_mode_matches(saved,current):
    if not saved or saved.get('format_version')!=1 or not all(saved.get(k) for k in ('dmd','slm')):
        raise ValueError('标定缺少 DMD/SLM 视频模式记录；旧文件保留，请在当前视频模式下重新标定后再用于实机')
    if not current or current.get('format_version')!=1:
        raise ValueError('当前视频模式读回不可用，停止实机操作')
    mismatches=[role.upper() for role in ('dmd','slm') if saved[role]!=current.get(role)]
    if mismatches:
        raise ValueError('视频模式改变（%s）；刷新率/信号尺寸变化可能导致输入偏移或截断。请恢复标定时模式，或重新标定；不能仅核对 600×600 画布' % ', '.join(mismatches))


def capture_display_mode(manager):
    getter=getattr(manager,'get_display_mode_signature',None)
    return getter() if callable(getter) else None


def validate_saved_display_mode(manager,calibration):
    saved=calibration.get('display_mode')
    current=capture_display_mode(manager)
    # Pure offline/test managers have no displays. Real HardwareManager always
    # provides a getter and fails closed for missing legacy metadata.
    if current is None and saved is None:return
    assert_display_mode_matches(saved,current)
