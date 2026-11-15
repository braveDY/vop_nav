

class NavigationVecEnvWrapper:
    def __init__(self, env):
        self.env = env.unwrapped
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.observation_manager.group_obs_dim["policy"][0]
        self.num_actions = self.env.action_manager.total_action_dim
        critic_shape = self.env.observation_manager.group_obs_dim.get("critic")
        self.num_privileged_obs = critic_shape[0] if critic_shape is not None else None
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        self.env.reset()

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def reset(self):
        observation, _ = self.env.reset()
        return observation["policy"], observation.get("critic")

    def get_observations(self):
        return self.env.obs_buf["policy"]

    def get_privileged_observations(self):
        return self.env.obs_buf.get("critic")

    def step(self, actions):
        observation, reward, terminated, truncated, extras = self.env.step(actions)
        done = terminated | truncated
        infos = dict(extras)
        infos["time_outs"] = truncated
        if "log" in extras and done.any():
            infos["episode"] = extras["log"]
        return observation["policy"], observation.get("critic"), reward, done, infos
