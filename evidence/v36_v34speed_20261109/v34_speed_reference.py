def faster_velocity(env):
    r = env.scene['robot']
    h = r.find_sites(['head'])[0][0]
    z = r.data.site_pos_w[:, h, 2] - env.scene.env_origins[:, 2]
    v = r.data.site_lin_vel_w[:, h, 2]
    s = rw._recovery_stage(env)
    target = z.new_tensor((0.62, 0.86, 1.15, 1.15))[s]
    desired = z.new_tensor((0.2, 0.3, 0.4, 0))[s] * ((target - z) / 0.2).clamp(0, 1)
    limit = z.new_tensor((0.45, 0.5, 0.55, 0.2))[s]
    return torch.exp(-20 * (v - desired).square() - 40 * (v.abs() - limit).clamp_min(0).square())

def overspeed(env):
    r = env.scene['robot']
    h = r.find_sites(['head'])[0][0]
    s = rw._recovery_stage(env)
    limit = r.data.site_lin_vel_w.new_tensor((0.45, 0.5, 0.55, 0.2))[s]
    return (r.data.site_lin_vel_w[:, h, 2].abs() - limit).clamp_min(0).square()