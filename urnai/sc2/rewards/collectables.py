from urnai.rewards.reward_base import RewardBase

MINERALS_PER_SHARD = 100


class CollectablesReward(RewardBase):

    def __init__(self):
        self.previous_state = None
        self.score = 0
        self.total_reward = 0

    def get(self, obs, default_reward, terminated, truncated) -> int:
        
        reward = 0

        if self.previous_state is not None:
            prev_player = self.previous_state.player
            curr_player = obs.player
            if curr_player.minerals > prev_player.minerals:
                reward = curr_player.minerals - prev_player.minerals
                reward /= MINERALS_PER_SHARD

        self.previous_state = obs
        self.total_reward += reward
        return reward
    
    def reset(self) -> None:
        self.previous_state = None
        self.score = 0
        self.total_reward = 0