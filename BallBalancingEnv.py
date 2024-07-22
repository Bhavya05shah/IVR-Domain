import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

class BallBalanceEnv(gym.Env):
    def __init__(self):
        super(BallBalanceEnv, self).__init__()
        self.position = 0.0
        self.velocity = 0.0
        self.dt = 0.02  # time step(interval at which the environment's state is updated)
        self.max_steps = 1000  # maximum number of steps
        self.gravity = 9.8  # gravity used to update the velocity
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # three discrete actions: 0 (left tilt), 1 (right tilt), 2 (no tilt)
        self.observation_space = spaces.Box(low=np.array([-1.0, -np.inf]), high=np.array([1.0, np.inf]), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.position = np.random.uniform(-1.0, 1.0)  # random initial position
        self.velocity = 0.0
        self.current_step = 0
        return np.array([self.position, self.velocity], dtype=np.float32), {}

    def step(self, action):
        if action == 0:  # leftward bend
            tilt = -1
        elif action == 1:  # rightward bend
            tilt = 1
        else:  # no bend needed
            tilt = 0

        # Calculating the acceleration due to gravity and the bend/tilt
        force = tilt * self.gravity
        acceleration = force  # assuming mass of the ball is 1
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

        reward = -np.abs(self.position)  # reward is negative of the absolute value of the position
        self.current_step += 1

        # Terminate the episode if the ball's position goes beyond (-1, 1)
        done = self.current_step >= self.max_steps or self.position < -1.0 or self.position > 1.0

        return np.array([self.position, self.velocity], dtype=np.float32), reward, done, False, {}

    def render(self):
        # Create a simple text-based visualization
        board_width = 20
        ball_position_display = int((self.position + 1) / 2 * board_width)
        board = ['-'] * (board_width + 1)  # +1 for borders

        # Place the ball symbol on the board to create a simulation of the ball balancing env
        if 0 <= ball_position_display < len(board):
            board[ball_position_display + 2] = 'O'

        # Printing the board for visualising
        print(''.join(board))

        print(f"Position: {self.position:.2f}, Velocity: {self.velocity:.2f}")

# Create and check the custom environment
env = BallBalanceEnv()
check_env(env, warn=True)

env = DummyVecEnv([lambda: BallBalanceEnv()])
model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.0003, batch_size=64, buffer_size=50000, learning_starts=1000, target_update_interval=500)
model.learn(total_timesteps=500000)

model_path = "DQN_BallBalanceEnv"
model.save(model_path)

eval_env = BallBalanceEnv()

# Rendering the trained model to evaluate
for _ in range(10):
    obs, _ = eval_env.reset()
    done = False
    while not done:
        eval_env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action)

eval_env.close()
env.close()
