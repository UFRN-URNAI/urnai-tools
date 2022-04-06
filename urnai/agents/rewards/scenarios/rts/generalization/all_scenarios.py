from urnai.agents.rewards.scenarios.rts.generalization.buildunits import \
    BuildUnitsGeneralizedRewardBuilder
from urnai.agents.rewards.scenarios.rts.generalization.collectables import \
    CollectablesGeneralizedRewardBuilder
from urnai.agents.rewards.scenarios.rts.generalization.defeatenemies import \
    DefeatEnemiesGeneralizedRewardBuilder
from urnai.agents.rewards.scenarios.rts.generalization.findanddefeat import \
    FindAndDefeatGeneralizedRewardBuilder


class MultipleScenarioRewardBuilder:
    SCENARIOS = {
        'GeneralizedCollectablesScenario': CollectablesGeneralizedRewardBuilder,
        'GeneralizedFindAndDefeatScenario': FindAndDefeatGeneralizedRewardBuilder,
        'GeneralizedDefeatEnemiesScenario': DefeatEnemiesGeneralizedRewardBuilder,
        'GeneralizedBuildUnitsScenario': BuildUnitsGeneralizedRewardBuilder,
    }

    def __init__(self, scenario):
        reward_builder_class = self.get_reward_builder(scenario)
        self.reward_builder = reward_builder_class()

    def get_reward_builder(self, scenario):
        for scenario_class in MultipleScenarioRewardBuilder.SCENARIOS.keys():
            if scenario == scenario_class:
                return MultipleScenarioRewardBuilder.SCENARIOS[scenario_class]

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return getattr(self.reward_builder, attr)(*args, **kwargs)

        return wrapper
