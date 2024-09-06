import unittest

from pysc2.lib import actions

from urnai.sc2.actions.sc2_action import SC2Action

_BUILD_REFINERY = actions.RAW_FUNCTIONS.Build_Refinery_pt
_NO_OP = actions.FUNCTIONS.no_op

class TestSC2Action(unittest.TestCase):

    def test_run(self):

        run_no_op = SC2Action.run(_NO_OP)
        run_build_refinery = SC2Action.run(_BUILD_REFINERY, 'now', 0)
        
        self.assertEqual(run_no_op.function, _NO_OP.id)
        self.assertEqual(run_no_op.arguments, [])

        self.assertEqual(run_build_refinery.function, _BUILD_REFINERY.id)
        self.assertEqual(run_build_refinery.arguments, [[actions.Queued['now']], [0]])
