"""Original-baseline selection only; no SDK, display, or recovery writes."""
from copy import deepcopy


def recovery_target(fault, failed_before, last_camera_report):
    if not fault:
        return None
    if failed_before:
        return deepcopy(failed_before)
    report = last_camera_report or {}
    before = report.get('camera_before')
    if (fault.startswith('显示核验失败')
            and report.get('camera_settings_restored') is True
            and before and report.get('camera_after') == before):
        return deepcopy(before)
    return None
