"""Startup-only two-page composition; importing this module performs no IO.

Detach legacy tabs, not their QObjects: the proven camera worker still sends
status/cleanup signals to those objects. No new camera owner or GUI subclass.
"""


def attach(window):
    if (getattr(window, 'raw_workbench', None) is not None
            or window.worker is not None or window.running
            or window.deployment_busy or window.adaptation_active or window.loop_active):
        raise RuntimeError('two-page composition requires one fresh, idle window')
    tabs = window.tabs
    flow = window.stability_workflow
    sampler = window.state_sampling_controller
    expected = [tabs.widget(0), window.deployment_page, window.adaptation_page,
                window.loop_page, sampler.widgets['page'], flow.widgets['page']]
    if tabs.count() != 6 or any(tabs.widget(i) is not page for i, page in enumerate(expected)):
        raise RuntimeError('unexpected page order; refuse partial UI removal')
    domain = flow.widgets['result_domain']
    raw_index = domain.findData('raw')
    if raw_index < 0:
        raise RuntimeError('RAW results unavailable')

    def disabled(*args, **kwargs):
        raise RuntimeError('legacy task entry removed; RAW head is not qualified for control')

    # Disconnect already-bound Qt callbacks as well as disabling Python entrypoints.
    signals = [(window.deployment_page, 'run_requested'),
               (window.deployment_page, 'freeze_requested'),
               (window.adaptation_page, 'collect_requested'),
               (window.adaptation_page, 'pilot_requested'),
               (window.loop_page, 'start_requested'),
               (window.loop_page, 'push_requested')]
    for page, name in signals:
        try:
            getattr(page, name).disconnect()
        except TypeError:  # Already disconnected by the original LoopWindow.
            pass
    for name in ('run_deployment', 'freeze_deployment', 'run_adaptation', 'run_loop', 'queue_push'):
        setattr(window, name, disabled)
    sampler.widgets['start'].clicked.disconnect()
    sampler.widgets['start'].setEnabled(False)
    # Keep sampler, its timer, worker callbacks, stop/blackout and rollback alive.
    # Hidden adaptation rendering is wasted work on every camera preview tick.
    window.adaptation_page.fresh_view = lambda *args, **kwargs: None
    window.consume_loop_view = lambda: None
    for i in (4, 3, 2, 1):
        tabs.removeTab(i)
        expected[i].hide()
        expected[i].setEnabled(False)
    tabs.setTabText(0, '手动对齐')
    tabs.setTabText(1, '八区稳定性测试 · RAW')
    tabs.setCurrentIndex(0)
    domain.setCurrentIndex(raw_index)
    domain.setToolTip('当前路线使用浮点RAW均值积分；非线性读出仅作历史对照。稳定性通过不等于控制部署通过。')
    for key in ('software_gain', 'black', 'gamma'):
        flow.widgets[key].setToolTip('只影响“均值后黑电平/γ”对照，不影响RAW积分，也不改相机。')
    from PyQt5 import QtWidgets
    replacements = {
        '独立N帧均值 → 固定黑电平/γ → 八区；不使用预览EMA，不推进MuJoCo。测试结束／取消后全0。':
            '主读出：独立N帧RAW均值 → 八区；黑电平/γ另存对照。不使用预览EMA，不推进MuJoCo；结束／取消后全0。',
        '读出软件增益': '对照分支软件增益',
        '读出黑电平 ADU': '对照分支黑电平 ADU',
        '读出 γ': '对照分支 γ',
    }
    for label in flow.widgets['page'].findChildren(QtWidgets.QLabel):
        if label.text() in replacements:
            label.setText(replacements[label.text()])
    note = QtWidgets.QLabel('当前主线：同图独立N帧RAW均值 → 固定八区积分。显示增益不进入读出；'
                           '旧非线性头禁止沿用。后续先验证新位置/速度，再适配18参数头；目前未放行闭环。')
    note.setWordWrap(True)
    # The original page is a scroll area; insert into its content layout.
    page = flow.widgets['page']
    content = page.widget() if isinstance(page, QtWidgets.QScrollArea) else page
    content.layout().insertWidget(0, note)
    window.raw_workbench = dict(detached_pages=tuple(expected[1:5]), note=note,
                               readout='RAW_mean_native8', control_qualified=False)
    window.setWindowTitle('实时光路对齐 v19 · 两页RAW工作台' +
                          (' · 真实器件' if window.hardware else ' · 离线预览'))
    return window.raw_workbench
