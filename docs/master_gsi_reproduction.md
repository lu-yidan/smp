# Master GSI reproduction versus deployment adaptation

Reference: master 0e67286; branch codex/master-gsi-reproduction. Original env, PPO config, GSI, reward, reset and termination source files are unchanged. Uses current server dependency runtime; historical launch overrides/hardware are not known.

| | master96 / GPU 0 | deploy93 / GPU 1 |
|---|---|---|
| Robot | mjlab stock G1 | RoboMimic deployment model/control |
| Actor | noisy96D, including base linear velocity | noisy93D, base linear velocity removed |
| Critic | 96D x10 history | same |
| Physics | 5ms x4, solver10/line search20 | 2ms x10, solver100/line search50 |
| Reset | original GSI windows, generated velocities/history | same mechanism on deployment model |
| GSI | pool4096, replace1024 every2400 control steps | same |
| Prior | pretrained_getup_f2s2.pt | same |
| Reward | (0.7 upward head velocity + 0.3 head height) x SMP, ws6 | same |
| Episode | 5s, original low-SMP / stood-up termination | same |
| PPO | lr1e-3 adaptive, gamma0.99, fixed action std0.30 | same |
| Noise/DR | original noise, friction, encoder bias, torso COM, pushes | retained; explicit foot-pair friction inherits randomized foot geom friction |

Both from random initialization, seed20260910,4096 environments,10000 PPO updates,save500,W&B tabletennis/smp. No actor checkpoint loaded.10000 is a common comparison budget, overriding master default30000. Different actor input sizes mean weights cannot be identical despite common seed.

No procedural fall-reset mixture, static pose bank, V6 prior, extra smoothing/standing reward, or removed baseline termination. Deployment adaptation changes observation and physical/control contract together, not a single-factor observation ablation. Existing training jobs on GPUs2–5 remain running.

Preflight: each16 environments /2 updates, full4096 GSI pool; explicitly trigger refresh at step2400 and assert pool head advances1024. Preflight weights/states are never used for full training. Original source files remain byte-identical to master. Baseline termination occurs after original25-step standing hold (~0.5s), not10s quiet standing; training counters are not fixed lying-recovery evaluation success.

Observational logs add head height,raw SMP,GSI pool head,termination fractions without changing reward/actions/dones. No new collision filter or safety termination is silently added to native master. Explicit deployment pairs would override geom DR, so a deployment-only startup event propagates the existing foot friction samples to the pairs.

## Continuous deployment recovery,20s (GPU2)

`deploy93_continuous20_10000` is a fresh-policy counterpart to GPU1 deploy93. Only change the episode/termination protocol using `--continuous-20s`:20 seconds, timeout or numerical instability only. No low-SMP termination; no stood-up termination. GSI pool/refresh, generated velocities/history, f2s2 prior, original task reward, fixed0.30 action std,lr1e-3,gamma0.99,noise and DR remain unchanged. No extra quiet-standing reward is introduced. Same seed and93D architecture allow identical initial actor/critic weights to the5s deploy93 arm.

Numerical instability means nonfinite qpos/qvel/qacc/qacc_warmstart/sensor data or absolute qpos/qvel>1000,qacc/warmstart>1e6. This is not a termination for ordinary falling or motion quality.

The comparison jointly changes horizon and early termination, so it tests the combined continued-recovery protocol rather than isolating those effects. Longer post-rise experience may help, but the unchanged task/prior does not explicitly prohibit small steps. Existing master96 and deploy93 five-second reference jobs remain. The two30-second flat93 E/F jobs were stopped at user request; the two20-second flat93 jobs remain.
