import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class SimpleRFEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Map size (22 × 16)
        self.map_w = 22
        self.map_h = 16

        # Observation shapes
        self.screen_shape = (12, 64, 64)
        self.minimap_shape = (2, 64, 64)
        self.nonspatial_shape = (11,)

        # Action: single scalar = coordinate index
        # 22 * 16 = 352 positions
        self.action_space = spaces.Discrete(self.map_w * self.map_h)

        # Observation:
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(0, 1, shape=self.screen_shape, dtype=np.float32),
            "minimap": spaces.Box(0, 1, shape=self.minimap_shape, dtype=np.float32),
            "nonspatial": spaces.Box(0, 1, shape=self.nonspatial_shape, dtype=np.float32),
        })

        # Internal positions (normalized 0–1)
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random player and target pos in [0,1]
        self.player_pos = np.random.rand(2)
        self.target_pos = np.random.rand(2)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Convert scalar into (x,y) map coordinate
        y = action // self.map_w
        x = action % self.map_w

        # Convert to normalized 0–1 coordinates
        target_x = x / (self.map_w - 1)
        target_y = y / (self.map_h - 1)
        target = np.array([target_x, target_y], dtype=np.float32)

        # Move player toward the chosen coordinate
        direction = target - self.player_pos
        dist = np.linalg.norm(direction) + 1e-8
        step_size = 0.06

        self.player_pos += step_size * direction / dist

        """
        _dir = (self.player_pos - self.target_pos)
        self.target_pos -= 0.01 * _dir / np.linalg.norm(_dir) + 1e-8
        self.target_pos[0] = np.clip(self.target_pos[0], 0, 1)
        self.target_pos[1] = np.clip(self.target_pos[1], 0, 1)
        """

        # Compute reward
        done = False
        reward = -0.01  # small step penalty

        # If touching target
        if np.linalg.norm(self.player_pos - self.target_pos) < 0.05:
            reward = 1.0
            done = True

        obs = self._get_obs()

        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    # ----------------------- OBSERVATION BUILDING ---------------------
    # ------------------------------------------------------------------

    def _get_obs(self):
        screen = self._render_layer(12)
        minimap = self._render_layer(2)
        nonspatial = self._build_nonspatial()

        return {
            "screen": screen,
            "minimap": minimap,
            "nonspatial": nonspatial,
        }

    def _render_layer(self, channels):
        """Creates a multi-channel 64x64 representation."""
        img = np.zeros((64, 64, channels), dtype=np.float32)

        px = int(self.player_pos[0] * 63)
        py = int(self.player_pos[1] * 63)
        tx = int(self.target_pos[0] * 63)
        ty = int(self.target_pos[1] * 63)

        # --- Draw on separate 2D scratch buffers (OpenCV compatible) ---

        # Player channel (0)
        buf0 = np.zeros((64, 64), dtype=np.float32)
        cv2.circle(buf0, (px, py), 3, 1.0, -1)
        img[:, :, 0] = buf0

        # Target channel (1)
        buf1 = np.zeros((64, 64), dtype=np.float32)
        cv2.rectangle(buf1, (tx - 3, ty - 3), (tx + 3, ty + 3), 1.0, -1)
        img[:, :, 1] = buf1

        # Remaining channels stay black

        return np.transpose(img, (2, 0, 1))

    def _build_nonspatial(self):
        """11 useful scalar features; fill extra with zeros."""
        dxdy = self.target_pos - self.player_pos
        dist = np.linalg.norm(dxdy)

        arr = np.zeros(11, dtype=np.float32)
        arr[0:2] = self.player_pos
        arr[2:4] = self.target_pos
        arr[4:6] = [0, 0]#dxdy
        arr[6] = 0#dist

        return arr

    # ------------------------------------------------------------------

    def render(self):
        return
        if self.render_mode != "human":
            return

        img = np.zeros((256, 256, 3), dtype=np.uint8)

        px = int(self.player_pos[0] * 255)
        py = int(self.player_pos[1] * 255)
        tx = int(self.target_pos[0] * 255)
        ty = int(self.target_pos[1] * 255)

        cv2.circle(img, (px, py), 8, (255, 100, 0), -1)     # player
        cv2.rectangle(img, (tx - 8, ty - 8), (tx + 8, ty + 8), (0, 0, 255), -1)  # target

        cv2.imshow("Simple RF Env", img)
        cv2.waitKey(1)

    def render_rgb(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        px = int(self.player_pos[0] * 255)
        py = int(self.player_pos[1] * 255)
        tx = int(self.target_pos[0] * 255)
        ty = int(self.target_pos[1] * 255)

        cv2.circle(img, (px, py), 8, (255, 100, 0), -1)
        cv2.rectangle(img, (tx - 8, ty - 8), (tx + 8, ty + 8), (0, 0, 255), -1)

        return img



# Register for Gymnasium
gym.envs.register(
    id="SimpleRFEnv-v0",
    entry_point=__name__ + ":SimpleRFEnv",
)

#import gymnasium as gym
#import simple_rf_env   # your file name

#env = gym.make("SimpleRFEnv-v0")

"""
obs, _ = env.reset()

for step in range(50):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    img = env.unwrapped.render_rgb()
    cv2.imwrite(f"data/simple_env/frame_{step}.png", img)
    if done:
        env.reset()

env.close()
"""