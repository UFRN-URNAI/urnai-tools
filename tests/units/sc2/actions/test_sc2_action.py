import unittest

from pysc2.lib import actions

from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions

_BUILD_REFINERY = actions.RAW_FUNCTIONS.Build_Refinery_pt
_NO_OP = actions.FUNCTIONS.no_op

class TestSC2Action(unittest.TestCase):

    def test_run(self):

        run_no_op = sc2_actions["no_op"].run()
        run_build_refinery = sc2_actions["Build_Refinery_pt"].run('now', 0)

        self.assertEqual(run_no_op.function, _NO_OP.id)
        self.assertEqual(run_no_op.arguments, [])

        self.assertEqual(run_build_refinery.function, _BUILD_REFINERY.id)
        self.assertEqual(run_build_refinery.arguments, [[actions.Queued['now']], [0]])
