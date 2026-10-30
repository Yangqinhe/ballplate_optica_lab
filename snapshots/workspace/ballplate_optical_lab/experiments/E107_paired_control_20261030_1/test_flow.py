"""Qt controller transitions with mocks only; no device window/camera owner."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest
import numpy as np
from page import Controller


def fixture():
    state=dict(time=.02,position=[0,0,.036],velocity_world=[0,0],angles=[0,0],filtered_action=[0,0],fallen=False)
    p=NS(go=True,one=False,busy=dict(id=7,op='advance'),pending_reset=None,
        observation={'id':4},pending_impulse=[0,0],stage='WORKER',inited=True,
        show_states=Mock(),prepare=Mock(),send=Mock(),pause=Mock(),lock=Mock(),fail=Mock(),
        note=Mock(),views=[dict(bars=Mock()),dict(bars=Mock())])
    response=dict(id=7,op='advance',ok=True,result=dict(states_after=[state,state],states_before=[dict(state,time=0),dict(state,time=0)],
        simulation_sums=[1]*64,real_sums=[2]*64))
    return p,response


class FlowTests(unittest.TestCase):
    def test_stale_response_does_nothing(self):
        p,r=fixture();r['id']=6;Controller.accept(p,r)
        self.assertIsNotNone(p.busy);p.show_states.assert_not_called();p.prepare.assert_not_called()
    def test_error_does_not_advance_or_prepare(self):
        p,r=fixture();r.update(ok=False,error='bad frame');Controller.accept(p,r)
        p.fail.assert_called_once_with('bad frame');p.prepare.assert_not_called()
    def test_continue_one_prepare_after_valid_step(self):
        p,r=fixture();Controller.accept(p,r);p.prepare.assert_called_once();self.assertIsNone(p.observation)
    def test_pause_inflight_finishes_no_new_step(self):
        p,r=fixture();p.go=False;Controller.accept(p,r)
        p.show_states.assert_called_once();p.prepare.assert_not_called();p.lock.assert_called_once_with(False)
    def test_one_step_calls_pause(self):
        p,r=fixture();p.one=True
        def pause():p.go=False
        p.pause=Mock(side_effect=pause);Controller.accept(p,r)
        p.pause.assert_called_once();p.prepare.assert_not_called()
    def test_reset_after_atomic_step(self):
        p,r=fixture();p.go=False;p.pending_reset=[.02,.01];Controller.accept(p,r)
        p.send.assert_called_once_with('reset',xy=[.02,.01]);p.prepare.assert_not_called()
    def test_fall_stops_no_extra_constraint_on_action(self):
        p,r=fixture();r['result']['states_after'][0]['fallen']=True;p.info=Mock()
        Controller.accept(p,r);p.pause.assert_called_once();p.prepare.assert_not_called()
    def test_pause_does_not_touch_unowned_manual_projection(self):
        p=NS(go=False,saved_input=None,burst=Mock(),zero=Mock(),busy=None,inited=False,stage='',lock=Mock(),info=Mock())
        Controller.pause(p);p.zero.assert_not_called();p.burst.cancel.assert_called_once()
    def test_pause_invalidates_burst_and_requests_black(self):
        p=NS(go=True,saved_input=None,burst=Mock(),zero=Mock(),busy={'id':1},inited=True,stage='',lock=Mock(),info=Mock())
        Controller.pause(p);self.assertFalse(p.go);p.zero.assert_called_once();p.burst.cancel.assert_called_once()
    def test_pause_prepared_observation_is_cancelled(self):
        p=NS(go=True,saved_input=None,burst=Mock(),zero=Mock(),busy=None,inited=True,observation={'id':1},
             stage='',lock=Mock(),info=Mock(),send=Mock())
        Controller.pause(p);p.send.assert_called_once_with('cancel');self.assertIsNone(p.observation)
    def test_reset_while_sampling_does_not_write_physics(self):
        p=NS(pause=Mock(),pending_reset=None,inited=True,busy={'id':1},send=Mock())
        Controller.reset(p,[.1,0]);self.assertEqual(p.pending_reset,[.1,0]);p.send.assert_not_called()
    def test_impulse_queued_without_simulation(self):
        p=NS(impulse=Mock(),angle=Mock(),info=Mock());p.impulse.value.return_value=.005;p.angle.value.return_value=90
        Controller.push(p);self.assertEqual(p.pending_impulse,[.005,90])

if __name__=='__main__':unittest.main()
