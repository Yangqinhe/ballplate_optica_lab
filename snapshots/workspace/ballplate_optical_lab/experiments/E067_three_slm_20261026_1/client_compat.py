"""External-client-only stale accessibility object classification.

No GUI source writes, target injection, hardware operations, or action retries.
The existing traversal already skips defunct objects, but some GLib bindings
omit the error domain from str(error). Consult the structured domain as well.
"""
import re

def defunct(error):
    message=str(error);domain=getattr(error,'domain',None)
    return bool((domain=='atspi_error' or 'atspi_error' in message) and
        re.search(r'object /org/a11y/atspi/accessible/[0-9]+ does not exist',message))

def install():
    from night_opt_20261020 import manual_io_v17
    manual_io_v17.defunct=defunct
    return dict(scope='this external client process only',
        change='structured GLib error domain plus exact missing accessibility object pattern',
        original_safe_nodes_unchanged=True,exact_target_uniqueness_guards_unchanged=True,
        target_GUI_modified=False,action_retry=False,source_files_modified=False)
