"""Fixed candidate sizes, equal 64-trial strata, inherited full 23s protocol."""
import train_table_specialist as run
import torch
original_config=run.config
original_setup=run.t.setup
def config(n,arm,evaluation,seed):
 assert evaluation
 cfg,agent=original_config(n,arm,evaluation,seed)
 cfg.events['gsi_reset'].params['bank_path']='outputs/table_purchase_20261007/bank.npz'
 return cfg,agent
def setup(env):
 first=not hasattr(env,'_tb_bank');original_setup(env)
 if first:
  assert env.num_envs%12==0
  count=env.num_envs//12;ix=torch.arange(env.num_envs,device=env.device)
  env._tb_family=ix//(4*count);env._tb_pose=(ix//(2*count))%2;env._tb_band=(ix//count)%2;env._tb_source=((ix%count)<count//4).long()
run.config=config;run.t.setup=setup
if __name__=='__main__':run.main()
