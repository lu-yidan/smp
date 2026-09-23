"""Native MuJoCo, deterministic 93D deployment-contract development evaluation.

No DDS/hardware. These are paired nominal-dynamics development trials, not a
heldout test or a safety certificate. Direct-lift clearance is a diagnostic.
"""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import mujoco
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from smp.recovery.clutter_benchmark import load_case, CONTROL, JOINT_NAMES, sha
from smp.recovery.scenes import robot_geoms

CONDITIONS = ['height','upright','knees','base_speed','angular_speed','joint_rms',
              'foot_speed','foot_load','other_contact','foot_width','support_motion']


def actor_from_checkpoint(path):
    torch.set_num_threads(1)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    actor = MLPModel(TensorDict({'actor':torch.zeros(1,93)},batch_size=[1]),
                     {'actor':['actor']},'actor',29,hidden_dims=[512,256,128],
                     activation='elu',obs_normalization=True,
                     distribution_cfg={'class_name':'GaussianDistribution','init_std':.3,
                                       'std_type':'scalar','learn_std':False})
    actor.load_state_dict(ckpt['actor_state_dict'],strict=True); actor.eval()
    # Native normalizer is preserved; no independently approximated normalization.
    infer = actor.as_onnx(verbose=False).eval()
    x = torch.randn(32,93,generator=torch.Generator().manual_seed(923))
    with torch.inference_mode():
        err = float((actor(TensorDict({'actor':x},batch_size=[32]))-infer(x)).abs().max())
    assert err < 1e-6, err
    return infer, {'checkpoint_sha256':sha(path),'iteration':int(ckpt['iter']),
                   'native_actor_parity_max_error':err,'observation_dim':93}


class Controller:
    def __init__(self, actor, entry):
        self.actor=actor; self.entry=entry.astype(np.float32).copy();self.last=np.zeros(29,np.float32);self.i=0
        self.default=np.array(CONTROL['default_joint_pos'],np.float32)
        self.scale=np.array(CONTROL['action_scale'],np.float32)
        self.kp=np.array(CONTROL['kps'],np.float32);self.kd=np.array(CONTROL['kds'],np.float32)
        self.limits=np.array(CONTROL['tau_limit'],np.float32)

    def run(self, d):
        q=d.qpos[7:36].astype(np.float32);dq=d.qvel[6:35].astype(np.float32)
        # MuJoCo freejoint angular qvel is expressed in the local body frame.
        rot=np.empty(9);mujoco.mju_quat2Mat(rot,d.qpos[3:7])
        gravity=-rot.reshape(3,3)[2]
        obs=np.concatenate((d.qvel[3:6],gravity,q-self.default,dq,self.last)).astype(np.float32)
        assert obs.shape==(93,) and np.isfinite(obs).all()
        with torch.inference_mode():raw=self.actor(torch.from_numpy(obs[None])).numpy()[0]
        if not np.isfinite(raw).all():raise FloatingPointError('nonfinite action')
        self.last=np.clip(raw,-CONTROL['clip_actions'],CONTROL['clip_actions'])
        target=self.default+self.scale*self.last
        alpha=min(1.,(self.i+1)/CONTROL['warmup_steps']);self.i+=1
        target=(1-alpha)*self.entry+alpha*target
        target=np.clip(target,q+(self.kd*dq-self.limits)/self.kp,q+(self.kd*dq+self.limits)/self.kp)
        return target,obs,raw


class GeometryMetrics:
    def __init__(self,m,meta):
        self.m=m;self.robot=set(robot_geoms(m));self.meta=meta
        self.pelvis=m.body('pelvis').id;self.torso=m.body('torso_link').id
        self.feet=[m.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
        self.foot_geoms=[{g for g in self.robot if m.geom(g).name.startswith(s+'_foot')} for s in ('left','right')]
        self.object_geoms={m.geom(g).id for o in meta['objects'] for g in o.get('overhead_geoms',o['geoms'])}
        self.probe=mujoco.MjData(m);self.jp=np.zeros((3,m.nv));self.jr=np.zeros_like(self.jp)
        self.force=np.zeros(6);self.velocity=np.zeros(6)
        self.knees=[JOINT_NAMES.index(n) for n in ('left_knee_joint','right_knee_joint')]

    def contacts(self,d):
        loads=np.zeros(2);support_z=np.zeros(2);other=0.;headforce=0.;motion=0.
        for i,c in enumerate(d.contact):
            a,b=int(c.geom1),int(c.geom2)
            if (a in self.robot)==(b in self.robot):continue
            robot=a if a in self.robot else b; external=b if robot==a else a
            mujoco.mj_contactForce(self.m,d,i,self.force)
            f=c.frame.reshape(3,3).T@self.force[:3]*(1 if robot==b else -1)
            foot=next((k for k,gs in enumerate(self.foot_geoms) if robot in gs),None)
            if foot is None:
                other+=abs(f[2])
                if 'head' in self.m.geom(robot).name:headforce+=np.linalg.norm(f)
            else:
                load=max(0.,float(f[2]));loads[foot]+=load;support_z[foot]+=load*c.pos[2]
                if load>2:
                    body=int(self.m.geom_bodyid[external])
                    mujoco.mj_jac(self.m,d,self.jp,self.jr,c.pos,body)
                    motion=max(motion,float(np.linalg.norm(self.jp@d.qvel)))
        zs=np.divide(support_z,loads,out=np.zeros(2),where=loads>2)
        return loads,float(np.mean(zs)),float(other),float(headforce),motion

    def measure(self,d):
        loads,support_z,other,headforce,motion=self.contacts(d)
        head=d.xpos[self.torso]+d.xmat[self.torso].reshape(3,3)@np.array([0,0,.43])
        z=head[2]-support_z;upright=d.xmat[self.pelvis].reshape(3,3)[2,2]
        speeds=[]
        for b in self.feet:
            mujoco.mj_jacBody(self.m,d,self.jp,self.jr,b);speeds.append(np.linalg.norm(self.jp@d.qvel))
        width=np.linalg.norm(d.xpos[self.feet[0],:2]-d.xpos[self.feet[1],:2])
        cond=np.array([z>=1.15,upright>=.93,abs(d.qpos[7:36][self.knees]).max()<.8,
                       np.linalg.norm(d.qvel[:3])<.15,np.linalg.norm(d.qvel[3:6])<.3,
                       np.sqrt(np.mean(d.qvel[6:35]**2))<.5,max(speeds)<.1,
                       min(loads)>20,other<20,.12<=width<=.45,motion<.15],bool)
        return cond,dict(head_height=float(z),upright=float(upright),loads=loads.tolist(),
                         other_contact_n=other,head_contact_n=headforce,support_z=support_z,
                         width=float(width),support_speed=motion)

    def lift_blocked(self,d,height):
        # Sample exact compound collisions, not a solid box around a hollow object.
        # Stop at head height 1.25m (at least 10cm); this is NOT escape reachability.
        p=self.probe;p.qpos[:]=d.qpos;p.qvel[:]=0
        maximum=max(.10,min(.65,1.25-height))
        for dz in np.arange(.01,maximum+.005,.01):
            p.qpos[2]=d.qpos[2]+dz
            mujoco.mj_kinematics(self.m,p);mujoco.mj_collision(self.m,p)
            for c in p.contact:
                a,b=int(c.geom1),int(c.geom2)
                if c.dist<-.0005 and ((a in self.robot and b in self.object_geoms) or (b in self.robot and a in self.object_geoms)):
                    return True
        return False


def run_case(case,actor,out,seconds=20.):
    started=time.monotonic();m,d,meta=load_case(case)
    assert m.nu==29 and abs(m.opt.timestep-.002)<1e-9
    assert [m.joint(int(j)).name for j in m.actuator_trnid[:,0]]==list(JOINT_NAMES)
    assert m.jnt_qposadr[m.actuator_trnid[:,0]].tolist()==list(range(7,36))
    assert np.allclose(m.jnt_actfrcrange[m.actuator_trnid[:,0],1],CONTROL['tau_limit'])
    ctrl=Controller(actor,d.qpos[7:36]);gm=GeometryMetrics(m,meta)
    initial=d.qpos.copy();target=ctrl.entry.copy();obs=np.zeros(93,np.float32);raw=np.zeros(29,np.float32)
    hold=best=0.;first_upright=None;first_clear=None;clear_age=0.;released=False;blocked=True
    peak=np.zeros((29,3));highload=np.zeros(29);states=[];taus=[];dqs=[];obses=[];raws=[];trace=[]
    path=0.;anchor=None;max_drift=0.;refall=False;fall_age=0.;stable_once=False;unsafe=None
    last_xy=d.qpos[:2].copy();condition_fail=np.zeros(len(CONDITIONS));peak_head=0.
    warning0=np.array([w.number for w in d.warning]);warning_ids=[int(getattr(mujoco.mjtWarning,n)) for n in ('mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL')]
    for step in range(round(seconds/.002)):
        dq_pre=d.qvel[6:35].copy()
        d.ctrl[:]=np.clip(ctrl.kp*(target-d.qpos[7:36])-ctrl.kd*dq_pre,-ctrl.limits,ctrl.limits)
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(d.warning[i].number>warning0[i] for i in warning_ids):
            unsafe='nonfinite_or_mujoco_numerical_warning';break
        tau=d.qfrc_actuator[6:35].copy()
        peak=np.maximum(peak,np.column_stack((abs(tau),np.maximum(abs(dq_pre),abs(d.qvel[6:35])),abs(tau*dq_pre))))
        highload+=(abs(tau)>.9*ctrl.limits)*.002
        taus.append(tau);dqs.append(dq_pre)
        if (step+1)%10:continue
        mujoco.mj_forward(m,d)
        cond,vals=gm.measure(d);t=(step+1)*.002
        peak_head=max(peak_head,vals['head_contact_n'])
        if (step+1)%50==0:blocked=gm.lift_blocked(d,vals['head_height'])
        clear_age=0. if blocked else clear_age+.02
        clear=clear_age>=.3-1e-8
        if clear and not released:
            first_clear=t;released=True;anchor=d.qpos[:2].copy()
        if released:
            path+=float(np.linalg.norm(d.qpos[:2]-last_xy));max_drift=max(max_drift,float(np.linalg.norm(d.qpos[:2]-anchor)))
        last_xy=d.qpos[:2].copy()
        upright=vals['head_height']>=1.15 and vals['upright']>=.93
        if upright and first_upright is None:first_upright=t
        fallen=vals['head_height']<.65 or vals['upright']<.5
        fall_age=fall_age+.02 if stable_once and fallen else 0.;refall |= fall_age>=.2-1e-8
        stable=bool(cond.all() and clear)
        hold=hold+.02 if stable else 0.;best=max(best,hold);stable_once |= best>=1.-1e-8
        condition_fail+=~cond
        target,obs,raw=ctrl.run(d)
        states.append(np.r_[d.qpos,d.qvel]);obses.append(obs);raws.append(raw)
        trace.append([t,vals['head_height'],vals['upright'],hold,float(blocked),clear_age,
                      *vals['loads'],vals['other_contact_n'],vals['support_z'],vals['width'],vals['support_speed']])
    name=meta['case_id'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(name+'.npz'),state=np.asarray(states),obs=np.asarray(obses),raw_action=np.asarray(raws),
                        diagnostics=np.asarray(trace),tau=np.asarray(taus,np.float32),dq=np.asarray(dqs,np.float32),
                        initial_qpos=initial,nq=m.nq,nv=m.nv,physics_dt=.002,control_dt=.02)
    result={'case_id':name,'family':meta['family'],'direction':meta['direction'],'layout':meta['layout'],'source':meta['source'],
            'scene_sha256':meta['model_sha256'],'reset_sha256':meta['reset_sha256'],
            'success_1s':best>=1-1e-8 and unsafe is None,'success_10s':best>=10-1e-8 and unsafe is None,
            'best_hold_s':best,'first_upright_s':first_upright,'first_direct_lift_clear_s':first_clear,
            'refall_after_stable_1s':bool(refall),'post_clear_path_m':path if released else None,
            'post_clear_max_drift_m':max_drift if released else None,'unsafe':unsafe,
            'peak_tau_nm':peak[:,0].tolist(),'peak_dq_rad_s':peak[:,1].tolist(),'peak_power_w':peak[:,2].tolist(),
            'above_90pct_torque_s':highload.tolist(),'head_contact_peak_50hz_n':peak_head,
            'condition_failure_fraction':dict(zip(CONDITIONS,(condition_fail/max(len(trace),1)).tolist())),
            'elapsed_wall_s':time.monotonic()-started,'completed_sim_s':len(taus)*.002}
    (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result
