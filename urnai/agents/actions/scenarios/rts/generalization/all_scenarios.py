from urnai.agents.actions.scenarios.rts.generalization.buildunits import \
    BuildUnitsDeepRTSActionWrapper, BuildUnitsStarcraftIIActionWrapper
from urnai.agents.actions.scenarios.rts.generalization.collectables import \
    CollectablesDeepRTSActionWrapper, CollectablesStarcraftIIActionWrapper
from urnai.agents.actions.scenarios.rts.generalization.defeatenemies import \
    DefeatEnemiesDeepRTSActionWrapper, DefeatEnemiesStarcraftIIActionWrapper
from urnai.agents.actions.scenarios.rts.generalization.findanddefeat import \
    FindAndDefeatDeepRTSActionWrapper, FindAndDefeatStarcraftIIActionWrapper


class MultipleScenarioActionWrapper:
    GAME_DRTS = 'drts'
    GAME_SC2 = 'sc2'

    METHOD_SINGLE = 'single'
    METHOD_MULTIPLE = 'multiple'

    SCENARIOS = {
        GAME_DRTS: {
            'GeneralizedCollectablesScenario': CollectablesDeepRTSActionWrapper,
            'GeneralizedFindAndDefeatScenario': FindAndDefeatDeepRTSActionWrapper,
            'GeneralizedDefeatEnemiesScenario': DefeatEnemiesDeepRTSActionWrapper,
            'GeneralizedBuildUnitsScenario': BuildUnitsDeepRTSActionWrapper,
        },
        GAME_SC2: {
            'GeneralizedCollectablesScenario': CollectablesStarcraftIIActionWrapper,
            'GeneralizedFindAndDefeatScenario': FindAndDefeatStarcraftIIActionWrapper,
            'GeneralizedDefeatEnemiesScenario': DefeatEnemiesStarcraftIIActionWrapper,
            'GeneralizedBuildUnitsScenario': BuildUnitsStarcraftIIActionWrapper,
        },
    }

    def __init__(self, scenario, game, method=METHOD_SINGLE):
        action_wrapper_class = self.get_action_wrapper(scenario, game)
        self.action_wrapper = action_wrapper_class()
        self.method = method
        if self.method == MultipleScenarioActionWrapper.METHOD_MULTIPLE:
            if game == MultipleScenarioActionWrapper.GAME_DRTS:
                action_wrapper_class = self.get_action_wrapper(
                    scenario,
                    MultipleScenarioActionWrapper.GAME_SC2)
                self.alternate_action_wrapper = action_wrapper_class()
            elif game == MultipleScenarioActionWrapper.GAME_SC2:
                action_wrapper_class = self.get_action_wrapper(
                    scenario,
                    MultipleScenarioActionWrapper.GAME_DRTS)
                self.alternate_action_wrapper = action_wrapper_class()

            self.actw_list = [self.action_wrapper, self.alternate_action_wrapper]

    def reset(self):
        self.switch_action_wrapper()

    def switch_action_wrapper(self):
        if self.method == MultipleScenarioActionWrapper.METHOD_MULTIPLE:
            if self.action_wrapper == self.actw_list[0]:
                self.action_wrapper = self.actw_list[1]
            else:
                self.action_wrapper = self.actw_list[0]

    def get_action_wrapper(self, scenario, game):
        return MultipleScenarioActionWrapper.SCENARIOS[game][scenario]

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return getattr(self.action_wrapper, attr)(*args, **kwargs)

        return wrapper
