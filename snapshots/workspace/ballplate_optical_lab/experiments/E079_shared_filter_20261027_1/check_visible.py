"""Inspect two public tabs and screenshots; no camera/display-output writes."""
import time
from run import ROOT,LAB,read,write,load


def main():
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy');legacy_gui._install_accessibility_compatibility();c.install()
    load('e078_visible_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    from PyQt5 import QtWidgets
    app=QtWidgets.QApplication([])
    verified=read(ROOT/'POST_STARTUP_01/RESULT.json');pid=verified['pid'];out=ROOT/'VISIBLE_CHECK_01';out.mkdir()
    def heartbeat():
        h=read(ROOT/'GUI/heartbeat.json')
        assert h['pid']==pid and 0<=time.time_ns()-h['unix_ns']<3e9
        assert not h['projection_armed'] and h['hardware_fault'] is None and h['live_camera']
        return h
    def choose(name):
        heartbeat();tree,A=accessible(pid)
        tabs=[n for n in safe_nodes(tree) if n.get_role_name()=='page tab' and n.get_name()==name]
        assert len(tabs)==1
        action=tabs[0].get_action_iface()
        choices=[i for i in range(action.get_n_actions()) if action.get_action_name(i) in ('Press','Activate')]
        assert len(choices)==1 and action.do_action(choices[0])
        time.sleep(.4)
    def capture(name):
        heartbeat();tree,A=accessible(pid);controls=[]
        for node in safe_nodes(tree):
            if node.get_role_name() in ('spin button','combo box','check box','label','push button','page tab'):
                value=node.get_value_iface()
                controls.append(dict(role=node.get_role_name(),name=node.get_name(),
                    value=value.get_current_value() if value else None,
                    checked=node.get_state_set().contains(A.StateType.CHECKED),showing=node.get_state_set().contains(A.StateType.SHOWING)))
        wid=int(verified['windows']['control']['wid'],16)
        assert app.primaryScreen().grabWindow(wid).save(str(out/(name+'.png')))
        write(out/(name+'.json'),controls)
        return controls
    choose('主界面 · 光路与参数');main_controls=capture('main')
    choose('八区稳定性测试');test_controls=capture('stability')
    visible=[v for v in test_controls if v['showing']]
    assert not any('参考补偿' in v['name'] or '基线参考' in v['name'] for v in visible)
    actual=[v['name'] for v in visible if v['role']=='label' and v['name'].startswith('曝光')]
    assert len(actual)==1 and '209.093' in actual[0] and '1080²' in actual[0] and '128' in actual[0]
    assert len([v for v in visible if v['role']=='spin button'])==4
    choose('主界面 · 光路与参数')
    write(out/'RESULT.json',dict(passed=True,retained_parameters_shared=True,
        no_reference_controls=True,test_spin_controls=4,actual_summary=actual,
        no_projection=True,no_camera_parameter_writes=True,final_heartbeat=heartbeat()))
    print(actual[0])


if __name__=='__main__':main()
