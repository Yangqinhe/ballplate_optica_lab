"""Explicit mock IPC + actual MuJoCo smoke. No physical optical interfaces."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from live_adapt_v22.core import Bundle, digest, fit_head, write_new
    from live_adapt_v22.test_adaptation import samples
    bundle = Bundle(a.bundle); a.output.mkdir(parents=True, exist_ok=False)
    head = fit_head(*samples())
    head.update(bundle=str(bundle.path), bundle_sha256=bundle.sha,
                phase_sha256=digest(bundle.data['SLM_RAW_GRAY_NO_LUT']),
                native_weights_sha256=digest(bundle.data['native_weights']),
                phase_mode='learned', real_hardware_features=False, MOCK_FOR_SOFTWARE_TEST_ONLY=True)
    path = a.output/'MOCK_HEAD_NOT_FOR_HARDWARE.json'; write_new(path, head)
    command = [sys.executable, str(PROJECT/'live_adapt_v22/pilot.py'), '--head', str(path.resolve()),
               '--head-sha256', digest(path), '--output', str((a.output/'MuJoCo').resolve()),
               '--seconds', '.06', '--mock-for-tests']
    queries = 0
    with (a.output/'stderr.log').open('xb') as err:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)
        try:
            for line in child.stdout:
                message = json.loads(line)
                if message['kind'] == 'optical_query':
                    assert message['sequence'] == queries
                    assert digest(message['dmd']) == message['dmd_sha256']
                    reply = dict(sequence=queries, dmd_sha256=message['dmd_sha256'],
                                 raw_integrals8=[450000.]*8, query_latency_s=.001)
                    child.stdin.write(json.dumps(reply)+'\n'); child.stdin.flush(); queries += 1
                elif message['kind'] == 'pilot_complete':
                    assert message['report']['source'] == 'mock'
                    assert message['report']['completed'] and not message['report']['error']
            if child.wait(timeout=10) != 0:
                raise RuntimeError('mock simulator child failed; see stderr.log')
            assert queries == 3
        finally:
            if child.poll() is None: child.terminate(); child.wait(timeout=3)
    write_new(a.output/'SOFTWARE_QA.json', dict(mock=True, hardware_accessed=False, queries=queries,
        MuJoCo_simulated_seconds=.06, IPC_verified=True, no_real_actuators=True,
        does_NOT_validate_real_noise_or_control=True))
    print(json.dumps(dict(mock_queries=queries, hardware_accessed=False, success=True)))


if __name__ == '__main__': main()
