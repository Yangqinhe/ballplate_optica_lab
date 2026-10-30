"""Offline MuJoCo trajectories + teacher labels. No camera or display imports."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=102022)
    a = p.parse_args()
    import numpy as np
    import torch
    import mujoco
    import runtime96 as rt
    from model96 import resize_binary
    from env.ballplate_env import BallPlateEnv
    from fourframe_teacher import centroids, four_time_basis
    from live_deploy_v20.protocol import PreparedBundle, pack_dmd
    from live_adapt_v22.core import digest, write_new, RULES
    torch.set_num_threads(2)
    previous = PreparedBundle(a.catalog); previous.verify()
    fixture = torch.load(rt.FIXTURES, map_location='cpu', weights_only=False)
    teacher_w = torch.tensor(fixture['metadata']['teacher_weights'])
    teacher_b = torch.tensor(fixture['metadata']['teacher_bias'])
    a.output.mkdir(parents=True, exist_ok=False)
    xml_path = a.output/'MODEL.xml'
    with xml_path.open('x') as stream:
        stream.write(fixture['model_xml'])
    rows = []; files = [Path(__file__).resolve(), PROJECT/'live_adapt_v22/core.py', xml_path,
                       rt.FIXTURES, previous.path, Path(previous.data['contract']),
                       Path(previous.data['SLM_RAW_GRAY_NO_LUT']), Path(previous.data['native_weights'])]
    import inspect
    files += [Path(inspect.getfile(BallPlateEnv)), Path(inspect.getfile(centroids)), PROJECT/'model96.py']
    files += [PROJECT/'live_adapt_v22'/name for name in ('worker.py', 'pilot.py')]
    rng = np.random.default_rng(a.seed)
    # Predeclare trajectory splits before rendering any history or acquiring data.
    assignments = np.array(['train']*12+['select']*4+['test']*4)
    rng.shuffle(assignments)
    for group in range(20):
        seed = a.seed+group
        xy = rng.uniform(-.10, .10, 2); velocity = rng.uniform(-.16, .16, 2)
        env = BallPlateEnv(seed=seed, xml=fixture['model_xml'], duration=1.)
        try:
            image = env.reset(xy)
            dof = int(env.model.jnt_dofadr[env.model.joint('ball_free').id])
            env.data.qvel[dof:dof+2] = velocity
            mujoco.mj_forward(env.model, env.data)
            history = deque([image.copy() for _ in range(4)], maxlen=4)
            for step in range(30):
                raw128 = np.stack(history)
                with torch.no_grad():
                    basis = four_time_basis(centroids(torch.from_numpy(raw128)[None]))
                    action = (basis@teacher_w.T+teacher_b)[0].numpy().clip(-.8, .8)
                if step in (4, 10, 18, 26):
                    raw96 = resize_binary(raw128)
                    index = len(rows); raw_path = a.output/f'input_{index:03d}_RAW96.npy'
                    dmd_path = a.output/f'input_{index:03d}_DMD.npy'
                    np.save(raw_path, raw96, allow_pickle=False)
                    np.save(dmd_path, pack_dmd(raw96), allow_pickle=False)
                    rows.append(dict(id=index, trajectory=group, split=str(assignments[group]),
                        seed=seed, trajectory_initial_xy_m=xy.tolist(),
                        trajectory_initial_velocity_m_s=velocity.tolist(), step=step,
                        simulated_time_s=step*env.control_dt, position_m=env.position().tolist(),
                        raw96=str(raw_path.resolve()), dmd=str(dmd_path.resolve()),
                        four_RAW128_unique_frames=int(len(np.unique(raw128.reshape(4, -1), axis=0))),
                        teacher_action_rad=action.tolist()))
                    files += [raw_path, dmd_path]
                image, _, fallen, _, _ = env.step(action)
                if fallen:
                    raise RuntimeError('teacher trajectory fell; do not silently replace it')
                history.append(image)
        finally:
            env.close()
        print(json.dumps(dict(prepared_trajectory=group, split=str(assignments[group]),
                              no_hardware=True)), flush=True)
    data = dict(schema='real-single-frame-adaptation-inputs-v22', seed=a.seed,
                created_unix_ns=time.time_ns(), cases=rows, rules=RULES,
                contract=previous.data['contract'], native_weights=previous.data['native_weights'],
                SLM_RAW_GRAY_NO_LUT=previous.data['SLM_RAW_GRAY_NO_LUT'],
                model_xml=str(xml_path.resolve()), inherited_catalog=str(previous.path),
                inherited_catalog_sha256=previous.catalog_sha,
                teacher_training_only=True, no_state_or_centroid_in_actor=True,
                temporal_inputs_s=[-.06, -.04, -.02, 0.], actor_parameters=18,
                CCD_history=0, physical_mask_updated=False, hardware_enabled=False,
                release_approved=False,
                bound_sha256={str(f.resolve()): digest(f) for f in files})
    # Inherit every frozen dependency, including current reference evidence.
    data['bound_sha256'].update(previous.data['bound_sha256'])
    write_new(a.output/'catalog.json', data)
    print('READY '+str((a.output/'catalog.json').resolve()), flush=True)


if __name__ == '__main__':
    main()
