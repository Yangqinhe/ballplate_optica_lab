"""NEW user-authorized volatile 1-bit VSYNC profile; original windows stay black.

Default is pure dry-run. No prior per-pattern LUT backup exists. Failure stops
the candidate; it does NOT pretend to restore the unknown former definition.
No flash/reset/mode/input/trigger-delay/SLM/network writes are implemented.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import stat
import struct
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from read_dlpc900_linux_v100 import codec, discover, read_packet, CODEC, CODEC_SHA256
from diagnose_black_restart_v83 import TimedBlackHold
from inspect_recovery_ui_v20 import inspect
from live_adapt_v22.core import digest, write_new

EXPOSURE_US = 7000
CONFIG = struct.pack('<HI', 1, 0)
KEYS = ('display_mode', 'input_source', 'port_clock', 'trigger_out1',
        'trigger_out2', 'pattern_lut_config', 'pattern_invert', 'block_load')
EXPORT = 'Video Mode\n0,1,7000,0,1,1,1,1\n'


def definition():
    # Clear after exposure; depth 1; colour 1; wait next VSYNC in Video Pattern.
    # TI GUI "Trigger Out 2 = 1" maps to wire disable-bit = 0.
    return struct.pack('<H', 0)+EXPOSURE_US.to_bytes(3, 'little')+b'\x91'+b'\0'*6


def write_packet(command, payload, sequence):
    allowed = {0x1A24: (b'\x00', b'\x02'),
               0x1A31: (CONFIG,), 0x1A34: (definition(),)}
    if (command not in allowed or payload not in allowed[command]
            or type(sequence) is not int or not 0 <= sequence <= 255):
        raise ValueError('outside exact new volatile profile write whitelist')
    return (struct.pack('<BBBHH', 0, 0x40, sequence, len(payload)+2, command)
            + payload).ljust(65, b'\0')


def validate_ack(raw, sequence):
    packet = bytes(raw)
    if len(packet) == 65 and packet[0] == 0:
        packet = packet[1:]
    if len(packet) < 4:
        raise ValueError('missing/truncated write ACK')
    flags, echoed, length = struct.unpack('<BBH', packet[:4])
    if flags not in (0, 0x40) or echoed != sequence or length > len(packet)-4:
        raise ValueError('NACK, wrong sequence, flags or truncated write ACK')
    return dict(flags=flags, sequence=echoed, length=length,
                data_hex=packet[4:4+length].hex())


def timing(refresh_hz):
    if not 119.95 <= refresh_hz <= 119.99:
        raise ValueError('unchanged approximately 119.972 Hz DMD signal required')
    period = 1e6/refresh_hz
    if period-EXPOSURE_US < 1000:
        raise ValueError('profile lacks declared >=1000 us VSYNC timing margin')
    return dict(video_refresh_hz=refresh_hz, frame_period_us=period,
                proposed_exposure_us=EXPOSURE_US, margin_us=period-EXPOSURE_US,
                measured_trigger_waveform=False, optical_window_qualified=False)


def plan():
    return dict(role='NEW_VOLATILE_SINGLEBIT_VSYNC_PROFILE_NOT_OLD_BACKUP',
        source_sha256=digest(__file__), frozen_codec_sha256=CODEC_SHA256,
        profile=dict(index=0, bit_position=0, bit_depth=1, exposure_us=EXPOSURE_US,
            dark_us=0, colour_flag=1, wait_next_VSYNC=True, clear_after_exposure=True,
            trigger_out2_enabled=True, entries=1, repeat_indefinitely=True),
        definition_hex=definition().hex(), lut_config_hex=CONFIG.hex(), GUI_export=EXPORT,
        old_per_pattern_definition_unknown_and_NOT_recoverable=True,
        user_accepted_replacement_of_unknown_volatile_sequence=True,
        authorization_source='current user continue after explicit volatile replacement question',
        maximum_seconds=120, failure_action='STOP only; never invent old LUT restoration',
        allowed_write_commands=['1A24 stop/start', '1A34 exact definition', '1A31 exact count'],
        no_flash_reset_mode_input_trigger_delays_SLM_network_or_new_windows=True,
        camera_SDK_opened=False, release_approved=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--accept-unknown-volatile-sequence-replacement', action='store_true')
    a = p.parse_args()
    prospective = plan()
    if not a.execute:
        print(json.dumps(dict(prospective, USB_opened=False, writes_sent=0), indent=2))
        return
    if not a.accept_unknown_volatile_sequence_replacement or a.session is None or a.output is None:
        p.error('explicit replacement acceptance, existing session and NEW output required')
    protocol = codec()
    devices = discover()
    if len(devices) != 1 or not devices[0]['user_can_read_write']:
        raise RuntimeError('one permitted DLPC900 interface 0 required')
    preflight = inspect(a.session, parked=True)
    prospective['video_timing'] = timing(preflight['roles']['dmd']['refresh_hz'])
    dependencies = {str(Path(__file__).resolve()): digest(__file__), str(CODEC): CODEC_SHA256}
    for name in ('read_dlpc900_linux_v100.py', 'diagnose_black_restart_v83.py',
                 'probe_line0_trigger_v18.py', 'inspect_recovery_ui_v20.py'):
        dependencies[str(PROJECT/'tools'/name)] = digest(PROJECT/'tools'/name)
    prospective.update(dependencies=dependencies, device=devices[0], created_unix_ns=time.time_ns())
    a.output.mkdir(parents=True, exist_ok=False)
    write_new(a.output/'PROSPECTIVE_PROFILE.json', prospective)
    write_new(a.output/'ORIGINAL_BLACK_PREFLIGHT.json', preflight)
    with (a.output/'NEW_SINGLEBIT_VSYNC_NOT_OLD_BACKUP.txt').open('x') as f:
        f.write(EXPORT)
    started = time.monotonic()
    guard = TimedBlackHold(a.session, preflight, started+120)
    result = dict(completed=False, error=None, transactions=[], candidate_retained=False,
        old_definition_restored=False, old_definition_unknown=True, new_sequence_may_have_changed=False,
        cleanup_stop_attempted=False, release_approved=False, camera_SDK_opened=False)
    fd = lock = None

    def verify(cleanup=False):
        # Cleanup can only STOP after expiry, still requiring same black windows,
        # USB identity and unchanged source hashes; never another START/definition.
        if cleanup:
            from probe_line0_trigger_v18 import BlackHold
            BlackHold.check(guard, force=True)
        else:
            guard.check(force=True)
        if discover() != devices or any(digest(k) != v for k, v in dependencies.items()):
            raise RuntimeError('USB identity/access or source changed')

    def transact(name, command=None, payload=None, cleanup=False):
        if cleanup and (command, payload) != (0x1A24, b'\0'):
            raise ValueError('cleanup permits STOP only')
        verify(cleanup)
        sequence = len(result['transactions'])+1
        if sequence > 255:
            raise RuntimeError('transaction sequence budget exceeded')
        packet = (read_packet(protocol, name, sequence) if command is None
                  else write_packet(command, payload, sequence))
        row = dict(name=name, operation='READ' if command is None else 'WRITE',
            sequence=sequence, request_hex=packet.hex(), cleanup=cleanup,
            requested_unix_ns=time.time_ns())
        result['transactions'].append(row)
        write_new(a.output/f'{sequence:03d}_INTENT.json', row)
        if command is not None:
            result['new_sequence_may_have_changed'] = True
        try:
            if os.write(fd, packet) != len(packet):
                raise RuntimeError('short HID transfer, no retries')
            row['sent'] = True
            if not select.select([fd], [], [], 1.5)[0]:
                raise TimeoutError('HID response deadline, no transaction retry')
            raw = os.read(fd, 65)
            row['response_hex'] = raw.hex()
            row['decoded'] = (protocol.decode_reply(name, protocol.parse_reply(name, sequence, raw))
                              if command is None else validate_ack(raw, sequence))
            verify(cleanup)
            return row['decoded']
        except BaseException as error:
            row['error'] = repr(error)
            raise
        finally:
            write_new(a.output/f'{sequence:03d}_RESULT.json', row)

    def wait_running(expected):
        end = time.monotonic()+4
        while time.monotonic() < end:
            row = transact('main_status')
            if row['sequencer_running'] == expected:
                return
            time.sleep(.1)
        raise RuntimeError('sequencer failed to reach requested state')

    try:
        lock = open(f'/run/user/{os.getuid()}/ballplate_dlpc900_readonly.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        fd = os.open(devices[0]['node'], os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        if not stat.S_ISCHR(os.fstat(fd).st_mode):
            raise RuntimeError('not a character device')
        discarded = []
        for _ in range(16):
            if not select.select([fd], [], [], 0)[0]:
                break
            discarded.append(os.read(fd, 65).hex())
        result['preexisting_reports_discarded'] = discarded
        before = {name: transact(name) for name in KEYS}
        result['before_readable_config'] = before
        status = transact('main_status')
        result['before_hardware_status'] = transact('hardware_status')
        if (before['display_mode']['mode'] != 'video_pattern'
                or before['pattern_lut_config'] != dict(entries=1, patterns_to_display=0, repeat_indefinitely=True)
                or before['pattern_invert']['inverted'] or not status['external_video_locked']
                or status['parked'] or status['video_frozen']):
            raise RuntimeError('current mode/LUT/noninverted/locked-video prerequisites failed')
        transact('stop', 0x1A24, b'\0')
        wait_running(False)
        transact('new_definition', 0x1A34, definition())
        transact('commit_single_repeat', 0x1A31, CONFIG)
        if transact('pattern_lut_config') != before['pattern_lut_config']:
            raise RuntimeError('single-repeat count readback mismatch')
        transact('start', 0x1A24, b'\x02')
        wait_running(True)
        begin = len(result['transactions'])
        for index in range(10):
            h = transact('hardware_status')
            m = transact('main_status')
            print(json.dumps(dict(sample=index, hardware=h, main=m)), flush=True)
            time.sleep(.5)
        result['health'] = protocol.assess_status_series(result['transactions'][begin:])
        result['after_readable_config'] = {name: transact(name) for name in KEYS}
        result['readable_config_preserved'] = result['after_readable_config'] == before
        if not result['health']['status_reads_passed'] or not result['readable_config_preserved']:
            raise RuntimeError('candidate health or preserved readable settings failed')
        verify()
        result.update(completed=True, candidate_retained=True)
    except BaseException as error:
        result['error'] = repr(error)
        if fd is not None and result['new_sequence_may_have_changed']:
            result['cleanup_stop_attempted'] = True
            try:
                transact('failure_cleanup_stop', 0x1A24, b'\0', cleanup=True)
                result['cleanup_stop_ACK_received'] = True
                result['candidate_retained'] = False
            except BaseException as cleanup_error:
                result['cleanup_error'] = repr(cleanup_error)
    finally:
        if fd is not None:
            os.close(fd)
        if lock is not None:
            lock.close()
        try:
            result['final_original_black'] = inspect(a.session, parked=True)
        except BaseException as error:
            result['final_black_error'] = repr(error)
            result['completed'] = False
        guard.close()
        result.update(wall_seconds=time.monotonic()-started, finished_unix_ns=time.time_ns(),
            writes_sent=sum(r.get('sent', False) for r in result['transactions'] if r['operation']=='WRITE'))
        write_new(a.output/'PROFILE_RESULT.json', result)
        print(json.dumps({k: result.get(k) for k in ('completed','error','health',
            'candidate_retained','writes_sent','cleanup_error')}, ensure_ascii=False), flush=True)
    if not result['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
