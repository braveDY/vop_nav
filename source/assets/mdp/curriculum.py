import torch


def navigation_curriculum(env, env_ids) -> dict[str, float]:
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.episode_length_buf[ids] > 0]
    terrain = env.scene.terrain
    if len(ids) and not getattr(env, 'freeze_curriculum', False):
        distance = env.command_manager.get_command('goal')[ids, :2].norm(dim=-1)
        terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + (distance < 0.5).long()
                                      - (distance > 2.0).long()).clamp(0, 9)
        terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return {'level': terrain.terrain_levels.float().mean().item()}
