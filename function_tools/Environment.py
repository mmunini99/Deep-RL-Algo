import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit


def CreateEnvironment(name):
  env = gym.make(name, render_mode="rgb_array")
  env = TimeLimit(env, max_episode_steps=1000)
  env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)
  env = RecordEpisodeStatistics(env)
  return env