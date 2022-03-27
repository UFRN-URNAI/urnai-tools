from urnai.agents.states.scenarios.rts.generalization.buildunits import \
    BuildUnitsGeneralizedStatebuilder
from urnai.agents.states.scenarios.rts.generalization.collectables import \
    CollectablesGeneralizedStatebuilder
from urnai.agents.states.scenarios.rts.generalization.defeatenemies import \
    DefeatEnemiesGeneralizedStatebuilder
from urnai.agents.states.scenarios.rts.generalization.findanddefeat import \
    FindAndDefeatGeneralizedStatebuilder
from urnai.utils.constants import RTSGeneralization


class MultipleScenarioStateBuilder:
    SCENARIOS = {
        'GeneralizedCollectablesScenario': CollectablesGeneralizedStatebuilder,
        'GeneralizedFindAndDefeatScenario': FindAndDefeatGeneralizedStatebuilder,
        'GeneralizedDefeatEnemiesScenario': DefeatEnemiesGeneralizedStatebuilder,
        'GeneralizedBuildUnitsScenario': BuildUnitsGeneralizedStatebuilder,
    }

    def __init__(self, scenario, method=RTSGeneralization.STATE_MAP):
        state_builder_class = self.get_state_builder(scenario)
        self.state_builder = state_builder_class(method=method)

    def get_state_builder(self, scenario):
        for scenario_class in MultipleScenarioStateBuilder.SCENARIOS.keys():
            if scenario == scenario_class:
                return MultipleScenarioStateBuilder.SCENARIOS[scenario_class]

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return getattr(self.state_builder, attr)(*args, **kwargs)

        return wrapper
