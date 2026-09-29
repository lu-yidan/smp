"""Install or verify the frozen A6 reset assets; stdlib hashes, optional NumPy/MuJoCo audit.

Use this helper from the current handoff branch, targeting a separate checkout of
A6_CODE. Installation refuses conflicting files and does not change training code.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

A6_CODE = '1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2'
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/recovery_study/assets/a6_reset_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_target(root, rel):
    path = Path(rel)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe asset path: {rel}')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes checkout: {rel}')
    return target


def install(root, assets, manifest):
    archive = assets / 'banks.tar.gz'
    blob = archive.read_bytes()
    if digest(blob) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 mismatch')
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name not in manifest['files'] or member.name in payload:
                raise ValueError(f'Unexpected/duplicate archive member: {member.name}')
            data = tar.extractfile(member).read()
            expected = manifest['files'][member.name]
            if len(data) != expected['bytes'] or digest(data) != expected['sha256']:
                raise ValueError(f'Corrupt archive member: {member.name}')
            target = checked_target(root, member.name)
            if target.exists() and (not target.is_file() or digest(target.read_bytes()) != expected['sha256']):
                raise ValueError(f'Refusing to overwrite a different asset: {target}')
            payload[member.name] = data
    if set(payload) != set(manifest['files']):
        raise ValueError('Incomplete archive')
    # All archive entries and all existing destinations are checked before writes.
    for name, data in payload.items():
        target = checked_target(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
    print(f'Installed/retained {len(payload)} exact files; no training code changed.')


def verify(root, manifest, allow_other_code=False):
    code = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if code != A6_CODE and not allow_other_code:
        raise ValueError(f'Expected A6 checkout {A6_CODE}, got {code}. Use --allow-other-code for asset-only auditing.')
    for name, expected in manifest['files'].items():
        data = checked_target(root, name).read_bytes()
        if len(data) != expected['bytes'] or digest(data) != expected['sha256']:
            raise ValueError(f'Bank/manifest mismatch: {name}')
    robot = root / 'src/smp/assets/deploy_g1'
    for name, expected in manifest['robot_files'].items():
        if digest(checked_target(robot, name).read_bytes()) != expected:
            raise ValueError(f'Robot asset mismatch: {name}')
    print(f"PASS: {len(manifest['files'])} bank/manifest files and {len(manifest['robot_files'])} model files. Code: {code}")


def audit(root, manifest, with_mujoco):
    import numpy as np
    report = {'banks': {}, 'historical_validation_only': True}
    group_sets = {}
    for name, expected in manifest['files'].items():
        if not name.endswith('.npz'):
            continue
        with np.load(root / name, allow_pickle=False) as bank:
            spec = expected['arrays']
            assert set(bank.files) == set(spec), name
            for key, schema in spec.items():
                assert list(bank[key].shape) == schema['shape'] and str(bank[key].dtype) == schema['dtype'], (name, key)
            q = bank['qpos']
            assert np.isfinite(q).all() and q.shape[1] == 36, name
            assert np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)) < 1e-5, name
            summary = {'n': len(q), 'qpos_columns': 36}
            for key in ('labels', 'stages', 'direction', 'stratum', 'source'):
                if key in bank.files:
                    values, counts = np.unique(bank[key], return_counts=True)
                    summary[key] = {str(v): int(c) for v, c in zip(values, counts)}
            if 'clips' in bank.files:
                groups = {str(s).replace('__mirror', '') for s in bank['clips']}
                group_sets[Path(name).stem] = groups
                summary['clip_groups'] = len(groups)
            report['banks'][name] = summary
    overlap = group_sets['train'] & group_sets['validation']
    assert not overlap, sorted(overlap)
    report['natural_train_validation_group_overlap'] = len(overlap)
    report['multiterrain_lineage_limit'] = 'No per-row clip IDs survive in this bank; no end-to-end independent final-test certification.'
    if with_mujoco:
        import mujoco
        robot = root / 'src/smp/assets/deploy_g1'
        tree = ET.parse(robot / 'source.xml').getroot()
        # Same robot-only extraction as historical deployment_robot_spec, without importing RL/GPU dependencies.
        tree.find('compiler').set('meshdir', str((robot / 'meshes').resolve()))
        order = [a.attrib['joint'] for a in tree.find('actuator')]
        for tag in ('actuator', 'sensor', 'contact', 'keyframe'):
            for item in tree.findall(tag):
                tree.remove(item)
        for world in tree.findall('worldbody'):
            for child in list(world):
                if child.tag != 'body' or child.get('name') != 'pelvis':
                    world.remove(child)
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
        assert model.nq == 36 and model.nv == 35
        assert [int(model.joint(n).qposadr[0]) for n in order] == list(range(7,36))
        report['robot_compile'] = {'mujoco': mujoco.__version__, 'nq': model.nq, 'nv': model.nv, 'nmesh': model.nmesh, 'joint_names': order, 'scope': 'CPU robot compile/order only, not a dynamics or RL parity test'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=DEFAULT)
    parser.add_argument('--allow-other-code', action='store_true')
    parser.add_argument('--audit', action='store_true', help='Inspect NPZ schema/splits; requires NumPy')
    parser.add_argument('--mujoco', action='store_true', help='Additionally compile the robot and check joint ordering')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((args.assets / 'manifest.json').read_text())
    if args.command == 'install':
        # Check target identity before writing anything.
        code = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if code != A6_CODE and not args.allow_other_code:
            raise ValueError('Install target must be the pinned historical A6 checkout')
        install(root, args.assets, manifest)
    verify(root, manifest, args.allow_other_code)
    if args.audit or args.mujoco:
        result = audit(root, manifest, args.mujoco)
        if args.report:
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
