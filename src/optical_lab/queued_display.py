"""Fresh-process-only mitigation for synchronous modulator repaint reentry.

No Qt/device imports or actions on import. Pixel ACK in the existing GUI stays
authoritative: image_applied acknowledges assignment, NOT physical presentation.
Historical Qt repaint ABRT is evidence for mitigation, not proof of the latest
crash's root cause. Never install this into an already-running output window.
"""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def queued_modulator(module):
    cls = module.OpticalModulator
    old_init, old_update = cls.__init__, cls.update_image
    own_repaint = cls.__dict__.get('repaint')

    def gui_thread(widget):
        if module.QtCore.QThread.currentThread() != widget.thread():
            raise RuntimeError('modulator repaint must stay on GUI thread')

    @wraps(old_init)
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        gui_thread(self)
        # Both synchronous repaint sites in the frozen implementation become
        # queued QWidget updates. No processEvents or nested painting here.
        self.label.repaint = self.label.update

    def repaint(self, *args):
        gui_thread(self)
        self.update(*args)

    @wraps(old_update)
    def update_image(self, image):
        # Worker-signal callers must not mutate a contiguous caller-owned buffer
        # after enqueueing it. QImage still makes its own copy in _apply_image.
        return old_update(self, module.np.array(image, dtype=module.np.uint8,
                                              order='C', copy=True))

    cls.__init__, cls.update_image, cls.repaint = init, update_image, repaint
    try:
        yield
    finally:
        cls.__init__, cls.update_image = old_init, old_update
        if own_repaint is None:
            del cls.repaint
        else:
            cls.repaint = own_repaint
