import gymnasium as gym

for task, config in [
    ('Unitree-Go2-VOP-BasicNav', 'Go2EnvCfg'),
    ('Unitree-Go2-VOP-BasicNav-Play', 'Go2PlayEnvCfg'),
]:
    gym.register(id=task, entry_point='envs.go2.nav_env:Go2NavigationEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'envs.go2.nav_env_cfg:' + config,
                         'train_cfg_entry_point': 'envs.go2.agents.ppo_cfg:AgentCfg'})

for task, config in [
    ('Unitree-Go2-VOP-Nav', 'VOPNavEnvCfg'),
    ('Unitree-Go2-VOP-Nav-Play', 'VOPNavPlayEnvCfg'),
]:
    gym.register(id=task, entry_point='envs.go2.vop_nav_env:VOPNavigationEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': 'envs.go2.vop_nav_env_cfg:' + config,
                         'train_cfg_entry_point': 'envs.go2.agents.vop_nav_ppo_cfg:VOPAgentCfg'})
