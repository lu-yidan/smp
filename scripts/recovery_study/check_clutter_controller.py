"""Audit observation/control contract, deployment actor fixtures and lift probe."""
import argparse,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
import yaml
from smp.recovery.clutter_rollout import actor_from_checkpoint,Controller,GeometryMetrics
from smp.recovery.clutter_benchmark import sha,load_case,CONTROL,JOINT_NAMES,lift_obstruction


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--deploy',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);report={};policies={}
    dep=yaml.safe_load((a.deploy/'policy/smp_recovery/config/smp_recovery.yaml').read_text())
    for key in ['default_joint_pos','action_scale','kps','kds','tau_limit','clip_actions','warmup_steps','control_dt']:assert np.array_equal(dep[key],CONTROL[key]),key
    for profile,fixture in [('R2_9000','ft_r2_9000'),('A6_9999','path_a6_9999'),('M2_2500',None)]:
        path=(a.out/'checkpoints'/(profile+'.pt')).resolve();actor,meta=actor_from_checkpoint(path)
        policies[profile]={'checkpoint':str(path),'sha256':meta['checkpoint_sha256']}
        if fixture:
            source=a.deploy/'outputs/recovery_candidates'/(fixture+'_parity.npz');x=np.load(source)
            with torch.inference_mode():pred=actor(torch.from_numpy(x['obs'])).numpy()
            meta['deployment_fixture_max_error']=float(abs(pred-x['actions']).max());assert meta['deployment_fixture_max_error']<1e-5
        report[profile]=meta
    report['control_yaml_numeric_parity']=True;report['geometry_probes']=[]
    for family in ['crossed_timber','ladder_boards','fixed_c_space','hollow_container']:
        case=next((a.root/'development').glob(family+'__prone__l0*'))
        m,d,meta=load_case(case);gm=GeometryMetrics(m,meta);before=d.qpos.copy()
        assert gm.lift_blocked(d,0.)
        assert lift_obstruction(m,d,meta)['all_objects_intersect_lift']
        assert np.array_equal(before,d.qpos)
        d.qpos[0]+=4;mujoco.mj_forward(m,d);assert not gm.lift_blocked(d,0.)
        report['geometry_probes'].append({'family':family,'blocked_then_clear':True,'probe_preserves_live_state':True})
    # Gravity and local angular velocity against independent native IMU sensor.
    rng=np.random.default_rng(93);m,d,_=load_case(case);ctrl=Controller(actor,d.qpos[7:36]);maxgyro=maxgravity=0.
    for i in range(32):
        q=rng.normal(size=4);q/=np.linalg.norm(q);d.qpos[3:7]=q;d.qvel[:]=rng.normal(size=m.nv);mujoco.mj_forward(m,d)
        target,obs,raw=ctrl.run(d)
        maxgyro=max(maxgyro,float(abs(obs[:3]-d.sensor('base_gyro').data).max()))
        gravity=-d.site_xmat[m.site('imu_in_pelvis').id].reshape(3,3)[2]
        maxgravity=max(maxgravity,float(abs(obs[3:6]-gravity).max()))
        expected=np.array(CONTROL['default_joint_pos'],np.float32)+np.array(CONTROL['action_scale'],np.float32)*np.clip(raw,-10,10)
        alpha=min(1,(i+1)/10);expected=(1-alpha)*ctrl.entry+alpha*expected
        qj=d.qpos[7:36].astype(np.float32);dq=d.qvel[6:35].astype(np.float32)
        expected=np.clip(expected,qj+(ctrl.kd*dq-ctrl.limits)/ctrl.kp,qj+(ctrl.kd*dq+ctrl.limits)/ctrl.kp)
        assert np.allclose(expected,target,atol=1e-6)
    assert maxgyro<1e-6 and maxgravity<1e-6
    report['imu_parity']={'angular_max_error':maxgyro,'gravity_max_error':maxgravity}
    # Four matched, obstacle-removed flat calibration cases; original fixtures unchanged.
    cal=a.out/'calibration';cal.mkdir(exist_ok=False)
    shutil.copytree(a.root/'assets',a.out/'assets',dirs_exist_ok=True)
    for case in sorted((a.root/'development').glob('crossed_timber__*__l0__v00')):
        original=json.loads((case/'manifest.json').read_text());tree=ET.parse(case/'scene.xml');xml=tree.getroot();world=xml.find('worldbody')
        for obj in original['objects']:world.remove(world.find(f"body[@name='{obj['name']}']"))
        for key in xml.find('keyframe'):
            key.set('qpos',' '.join(key.get('qpos').split()[:36]));key.set('qvel',' '.join(['0']*35))
        name='flat_calibration__'+original['direction'];folder=cal/name;folder.mkdir()
        (folder/'scene.xml').write_text(ET.tostring(xml,encoding='unicode'))
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
        np.savez_compressed(folder/'reset.npz',qpos=d.qpos,qvel=d.qvel)
        meta={**original,'case_id':name,'family':'flat_calibration','objects':[],'model_sha256':sha(folder/'scene.xml'),'reset_sha256':sha(folder/'reset.npz'),'paired_clutter_case':original['case_id']}
        (folder/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.out/'policies.json').write_text(json.dumps(policies,indent=2)+'\n')
    (a.out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
