"""Run on the current A6 source before launching the archived V34 bridge."""
import pickle
from train_a6_paper_ablation import build_config
cfg,_=build_config(128,'A6_no_alpha')
with open('a6_reward_bundle.pkl','wb') as f:
 pickle.dump(dict(rewards=cfg.rewards,sensors=[s for s in cfg.scene.sensors if s.name in ('quality_feet','quality_other','path_hands')],metrics={k:v for k,v in cfg.metrics.items() if k in ('v33_substep','v33_cache','r2_load_substep','r2_standing')}),f)
