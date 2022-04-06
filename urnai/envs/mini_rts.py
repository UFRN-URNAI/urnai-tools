import argparse
from os.path import expanduser
import sys

from urnai.utils.reporter import Reporter as rp


class MiniRTSEnv:

    # TODO: add enemy AI to environment, check: MIGHT HAVE BEEN DONE, NEED TO BE TESTED
    # https://github.com/facebookresearch/ELF/blob/master/train_minirts.sh and
    # https://github.com/facebookresearch/ELF/blob/master/eval_minirts.sh and
    # https://github.com/facebookresearch/ELF/blob/master/eval.py
    # TODO: TEST THIS!!!!!

    def __init__(self, elf_path=expanduser('~') + '/ELF', enemyAI=True, ai_fskip=50, model_fskip=50,
                 gpu=False, num_games=1, max_ticks=1000):
        """
        Initizalize attributes

        gpu = True #needs CUDA

        rts usage: game.py [-h] [--handicap_level HANDICAP_LEVEL] [--players PLAYERS]
        [--max_tick MAX_TICK] [--shuffle_player]
        [--num_frames_in_state NUM_FRAMES_IN_STATE]
        [--max_unit_cmd MAX_UNIT_CMD] [--seed SEED] [--actor_only]
        [--model_no_spatial]
        [--save_replay_prefix SAVE_REPLAY_PREFIX]
        [--output_file OUTPUT_FILE]
        [--cmd_dumper_prefix CMD_DUMPER_PREFIX] [--gpu GPU]
        [--use_unit_action] [--disable_time_decay]
        [--use_prev_units] [--attach_complete_info]
        [--feature_type FEATURE_TYPE] [--num_games NUM_GAMES]
        [--batchsize BATCHSIZE] [--game_multi GAME_MULTI] [--T T]
        [--eval] [--wait_per_group]
        [--num_collectors NUM_COLLECTORS] [--verbose_comm]
        [--verbose_collector] [--mcts_threads MCTS_THREADS]
        [--mcts_rollout_per_thread MCTS_ROLLOUT_PER_THREAD]
        [--mcts_verbose]
        [--mcts_save_tree_filename MCTS_SAVE_TREE_FILENAME]
        [--mcts_verbose_time] [--mcts_use_prior]
        [--mcts_pseudo_games MCTS_PSEUDO_GAMES]
        [--mcts_pick_method MCTS_PICK_METHOD]
        [--additional_labels ADDITIONAL_LABELS]
        """
        raise Exception(
            'Mini-RTS is not working yet. Game freezes. Until we work around it, this is'
            'exception is raised.')

        # import ELF's rlpytorch
        sys.path.append(elf_path)
        from rlpytorch.model_loader import load_module

        minirts_path = elf_path + '/rts/game_MC/game'
        game = load_module(minirts_path).Loader()

        # This might add an AI to the game
        parser = argparse.ArgumentParser()
        cmd = ['--num_games', str(num_games), '--batchsize', '128', '--players',
               "'fs=" + str(model_fskip) + ',type=AI_NN;fs=' + str(ai_fskip) + ",type=AI_SIMPLE'",
               '--additional_labels', 'id,last_terminal,seq', '--shuffle_player',
               '--num_frames_in_state', '1', '--max_tick', str(max_ticks)]

        if gpu:
            cmd += ['--gpu', '0']

        game.args.Load(parser, game, cmd_line=cmd)

        self.rts = game.initialize()

        self.observation = None
        self.actor_action = None
        self.done = True
        self.chosen_action = self._act_callback
        self.dummy_train = self._dummy_train_function

    def start(self):
        """
        Register action callback
        Start minirts
        """
        self.rts.reg_callback('actor', self.chosen_action)
        self.rts.reg_callback('train', self.dummy_train)
        self.rts.Start()
        self.done = False

    def step(self, action):
        """Executes action and return observation, done."""
        self.actor_action = action
        self.rts.Run()

        return self.observation, self.done

    def close(self):
        """Stops minirts."""
        self.rts.Stop()
        self.done = True

    def reset(self):
        """
        Restart environment and returns initial observation.
        TODO: check if self.start() modifies self.observation value
        """
        self.close()
        self.observation = None
        self.actor_action = None
        self.start()

        return self.observation

    def restart(self):
        self.reset()

    def _act_callback(self, batch):
        self.observation = batch
        return self.actor_action

    def _dummy_train_function(self, batch):
        return


rp.report('URNAI: start minirts')
rts = MiniRTSEnv()
rp.report('URNAI: run minirts')
rts.rts.Run()
rp.report('URNAI: closing minirts')
rts.close()
