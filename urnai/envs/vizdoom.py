from vizdoom import AutomapMode, Button, DoomGame, GameVariable, ScreenFormat, \
    ScreenResolution, UnsupportedVizDoomRes

from .base.abenv import Env
from ..utils.error import WadNotFoundError


class VizdoomEnv(Env):
    RES_640X480 = ScreenResolution.RES_640X480
    RES_320X240 = ScreenResolution.RES_320X240
    RES_160X120 = ScreenResolution.RES_160X120

    def __init__(self, wad, doommap='map01', _id='vizdoom', render=True, reset_done=True,
                 num_steps=1000, res=RES_160X120, auto_map=True):
        if wad is not None and wad != '':
            super().__init__(_id, render, reset_done)
            self.num_steps = num_steps
            self.wad = wad
            self.doommap = doommap
            self.game = DoomGame()
            self.auto_map = auto_map
            self.pickle_black_list.append('wad')

            self.res = res
            if self.res == VizdoomEnv.RES_160X120:
                self.res_w = 160
                self.res_h = 120
            elif self.res == VizdoomEnv.RES_320X240:
                self.res_w = 320
                self.res_h = 240
            elif self.res == VizdoomEnv.RES_640X480:
                self.res_w = 640
                self.res_h = 480
            else:
                raise UnsupportedVizDoomRes(
                    self.res + ' is an unsupported vizdoom resolution, use only'
                    + ' VizdoomEnv.RES_640X480, VizdoomEnv.RES_320X240 or VizdoomEnv.RES_160X120.')

        else:
            raise WadNotFoundError('A wad file is needed.')

    def start(self):
        """Start vizdoom."""
        # Init object, set wad and map
        self.game.set_doom_scenario_path(self.wad)
        if self.doommap is not None:
            self.game.set_doom_map(self.doommap)

        # Set available controls
        self.game.add_available_button(Button.ATTACK)
        self.game.add_available_button(Button.USE)
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.MOVE_BACKWARD)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.TURN_LEFT)

        # Set which variables will be set in observation
        # self.game.get_state().game_variables will be a nparray
        # where every position will be one of the variables
        # below. All variables are floats.
        # The list is ordered on the same order the variable
        # are added here
        self.game.add_available_game_variable(GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(GameVariable.ITEMCOUNT)
        self.game.add_available_game_variable(GameVariable.SECRETCOUNT)
        self.game.add_available_game_variable(GameVariable.DEATHCOUNT)
        self.game.add_available_game_variable(GameVariable.HITCOUNT)
        self.game.add_available_game_variable(GameVariable.HITS_TAKEN)
        self.game.add_available_game_variable(GameVariable.DAMAGECOUNT)
        self.game.add_available_game_variable(GameVariable.DAMAGE_TAKEN)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.ARMOR)
        self.game.add_available_game_variable(GameVariable.DEAD)
        self.game.add_available_game_variable(GameVariable.ATTACK_READY)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)

        # Set mini_map available in game obs
        # To get automap: observation.automap_buffer
        # Example: https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/automap.py
        if self.auto_map:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)

        # To get gamescreen on state, you should:
        # game.get_state().screen_buffer
        # Some graphic properties
        # screen format may be:
        # game.set_screen_format(ScreenFormat.RGB24)
        # game.set_screen_format(ScreenFormat.RGBA32)
        # game.set_screen_format(ScreenFormat.ARGB32)
        # game.set_screen_format(ScreenFormat.BGRA32)
        # game.set_screen_format(ScreenFormat.ABGR32)
        # game.set_screen_format(ScreenFormat.GRAY8)
        # self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(self.res)
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_window_visible(self.render)

        # Set reward -1 for every movement the agent makes
        self.game.set_living_reward(-1)

        # Set episode limit and a little delay to spawn monsters
        self.game.set_episode_timeout(self.num_steps)
        self.game.set_episode_start_time(0)

        self.game.init()

    def step(self, action):
        """Executes action and return observation, reward and done."""
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        state = self.game.get_state()
        if state is not None:
            self.observation = state

        return self.observation, reward, done

    def reset(self):
        """Resets the game to a new episode."""
        self.close()
        self.start()
        if self.game.get_state() is not None:
            self.observation = self.game.get_state()
        return self.observation

    def close(self):
        """Stops vizdoom."""
        if self.game is not None:
            self.game.close()

    def restart(self):
        self.reset()

    def get_screen_width(self):
        return self.res_w

    def get_screen_height(self):
        return self.res_h
