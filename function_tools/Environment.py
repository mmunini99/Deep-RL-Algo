import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit

def CreateEnvironment(name):
  env = gym.make(name, render_mode="rgb_array")
  env = TimeLimit(env, max_episode_steps=1000)
  env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
  env = RecordEpisodeStatistics(env)
  return env


class RepeatActionWrapper(gym.Wrapper):
  def __init__(self, env, n):
    super().__init__(env)
    self.env = env
    self.n = n

  def step(self, action):
    done = False
    total_reward = 0.0
    for _ in range(self.n):
      next_state, reward, done, trunc, info = self.env.step(action)
      total_reward += reward
      if done or trunc:
        break
    return next_state, total_reward, done, trunc, info

def CreateEnvironmentContinuous(name, time_repetition):
  env = gym.make(name, render_mode="rgb_array")
  env = TimeLimit(env, max_episode_steps=1000)
  env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
  env = RepeatActionWrapper(env, n = time_repetition)
  env = RecordEpisodeStatistics(env)
  return env