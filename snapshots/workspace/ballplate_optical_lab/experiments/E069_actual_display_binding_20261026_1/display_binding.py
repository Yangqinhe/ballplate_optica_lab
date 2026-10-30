"""Fresh-process display composition, targeting the actual Bench factory.

No SDK, Qt or display IO on import. The historical loader creates a SECOND
module from the same modulator.py file: equal filenames do not imply equal
Python classes. Patch both identities, and verify every actual output instance.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import sys
from pathlib import Path


@contextmanager
def bound_modulators(canonical, original_context, source, digest, evidence):
    bench = sys.modules.get('bench')
    if bench is None:
        raise RuntimeError('Bench factory not imported before display binding')
    actual = bench.modulator
    for module in (canonical, actual):
        path = Path(module.__file__).resolve()
        if path != source or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('actual output module source differs')
    evidence.update(canonical_module=canonical.__name__, actual_module=actual.__name__,
                    distinct_classes=canonical.OpticalModulator is not actual.OpticalModulator,
                    source=str(source), source_sha256=digest, instances_verified=False)
    with ExitStack() as stack:
        seen = set()
        for module in (canonical, actual):
            if id(module.OpticalModulator) not in seen:
                seen.add(id(module.OpticalModulator))
                stack.enter_context(original_context(module))
        evidence['actual_class'] = actual.OpticalModulator
        evidence['patched_update'] = actual.OpticalModulator.update_image
        evidence['patched_repaint'] = actual.OpticalModulator.repaint
        yield


def verify_instances(window, evidence):
    """Fail before camera start / projection if a different factory is used."""
    cls = evidence['actual_class']
    verified = {}
    for role, widget in window.windows.items():
        if (type(widget) is not cls or
                widget.update_image.__func__ is not evidence['patched_update'] or
                widget.repaint.__func__ is not evidence['patched_repaint'] or
                widget.label.repaint != widget.label.update):
            raise RuntimeError('output widget missed queued-display binding: ' + role)
        verified[role] = dict(class_module=type(widget).__module__,
                              queued_repaint=True, owned_signal_array=True)
    if window.hardware and set(verified) != {'dmd', 'slm'}:
        raise RuntimeError('two actual output instances required')
    return {k: v for k, v in evidence.items()
            if k not in ('actual_class', 'patched_update', 'patched_repaint')} | dict(
                instances=verified, instances_verified=bool(verified),
                hardware=window.hardware, native_root_cause_fully_proven=False)
