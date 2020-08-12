import os

import numpy as np
import pickle
import gzip
import re
import random

from fairmotion.utils import utils
from fairmotion.utils import conversions
from fairmotion.processing import operations
from fairmotion.motion.motion import Motion
from fairmotion.motion.velocity import MotionWithVelocity
from fairmotion.data import bvh

import env_humanoid_base

def load_motions(motion_files, skel, char_info, verbose):
    assert motion_files is not None
    motion_file_names = []
    for names in motion_files:
        head, tail = os.path.split(names)
        motion_file_names.append(tail)
    if isinstance(motion_files[0], str):
        motion_dict = {}
        motion_all = []
        for i, file in enumerate(motion_files):
            ''' If the same file is already loaded, do not load again for efficiency'''
            if file in motion_dict:
                m = motion_dict[file]
            else:
                if file.endswith('bvh'):
                    m = bvh.load(motion=Motion(name=file, skel=skel),
                                 file=file,
                                 scale=1.0, 
                                 load_skel=False,
                                 v_up_skel=char_info.v_up, 
                                 v_face_skel=char_info.v_face, 
                                 v_up_env=char_info.v_up_env)
                    m = MotionWithVelocity.from_motion(m)
                elif file.endswith('bin'):
                    m = pickle.load(open(file, "rb"))
                elif file.endswith('gzip') or file.endswith('gz'):
                    with gzip.open(file, "rb") as f:
                        m = pickle.load(f)
                else:
                    raise Exception('Unknown Motion File Type')
                if verbose: 
                    print('Loaded: %s'%file)
            motion_all.append(m)
    elif isinstance(motion_files[0], MotionWithVelocity):
        motion_all = motion_files
    else:
        raise Exception('Unknown Type for Reference Motion')

    return motion_all, motion_file_names

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)
        
        self._initialized = False
        self._config = config
        self._ref_motion = None
        self._imit_window = [0.05, 0.15]
        self._start_time = 0.0

        if config.get('lazy_creation'):
            if self._verbose:
                print('The environment was created in a lazy fashion.')
                print('The function \"create\" should be called before it')
            return

        self.create()

    def create(self):
        project_dir      = self._config['project_dir']
        ref_motion_db    = self._config['character'].get('ref_motion_db')
        ref_motion_scale = self._config['character'].get('ref_motion_scale')
        ref_motion_file  = []

        for i, mdb in enumerate(ref_motion_db):
            motions = []
            if mdb.get('cluster_info'):
                ''' Read reference motions based on the cluster labels '''
                assert mdb.get('data') is None, \
                    'This should not be specified when cluster_info is used'
                dir = mdb['cluster_info'].get('dir')
                label_file = mdb['cluster_info'].get('label_file')
                sample_id = mdb['cluster_info'].get('sample_id')
                labels = {}
                assert label_file
                if project_dir:
                    label_file = os.path.join(project_dir, label_file)
                with open(label_file, 'r') as file:
                    for line in file:
                        l = re.split('[\t|\n|,|:| ]+', line)
                        id, rank, score, filename = int(l[0]), int(l[1]), float(l[2]), str(l[3])
                        if id not in labels.keys(): labels[id] = []
                        labels[id].append({'rank': rank, 'socre': score, 'filename': filename})
                num_cluster = len(labels.keys())
                for j in range(num_cluster):
                    if sample_id and j!=sample_id:
                        continue
                    for label in labels[j]:
                        if project_dir:
                            file = os.path.join(project_dir, dir, label['filename'])
                        motions.append(file)
            else:
                ''' Read reference motions from the specified list of files and dirs '''
                ref_motion_data = mdb.get('data')
                motions = []
                if ref_motion_data.get('file'):
                    motions += ref_motion_data.get('file')
                if ref_motion_data.get('dir'):
                    for d in ref_motion_data.get('dir'):
                        if project_dir:
                            d = os.path.join(project_dir, d)
                        motions += utils.files_in_dir(d, ext=".bvh", sort=True)
                if project_dir:
                    for j in range(len(motions)):
                        motions[j] = os.path.join(project_dir, motions[j])
            ''' 
            If num_sample is specified, we use only num_sample motions 
            from the entire reference motions. 
            'random' chooses randomly, 'top' chooses the first num_sample
            '''
            num_sample = mdb.get('num_sample')
            if num_sample:
                sample_method = mdb.get('sample_method')
                if sample_method == 'random':
                    motions = random.choices(motions, k=num_sample)
                elif sample_method == 'top':
                    motions = motions[:num_sample]
                else:
                    raise NotImplementedError
            ref_motion_file.append(motions)
        
        ''' Load Reference Motion '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names = \
                load_motions(ref_motion_file[i], 
                             self._base_motion[i].skel,
                             self._sim_agent[i]._char_info,
                             self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Imitation Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, len(self.state(i)), self._action_space[i].dim))
            print('-------------------------------')

    def callback_reset_prev(self, info):
        
        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion = self.sample_ref_motion()
        
        ''' Choose a start time for the current reference motion '''
        
        start_time = info.get('start_time')
        if start_time is not None:
            self._start_time = start_time
        else:
            self._start_time = \
                np.random.uniform(0.0, self._ref_motion[0].length())

    def callback_reset_after(self, info):
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._init_poses[i], self._init_vels[i])

    def callback_step_after(self):
        ''' This is necessary to compute the reward correctly '''
        cur_time = self.get_current_time()
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._ref_motion[i].get_pose_by_time(cur_time),
                self._ref_motion[i].get_velocity_by_time(cur_time))

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time,
                  self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')
    
    def compute_init_pose_vel(self, info):
        '''
        This performs reference-state-initialization (RSI)
        '''
        init_poses, init_vels = [], []
        cur_time = self.get_current_time()

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''
            cur_pose = self._ref_motion[i].get_pose_by_time(cur_time)
            cur_vel = self._ref_motion[i].get_velocity_by_time(cur_time)
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                cur_pose, cur_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], cur_pose, cur_vel)
            init_poses.append(cur_pose)
            init_vels.append(cur_vel)
        return init_poses, init_vels
    
    def state_body(self, idx):
        return self._state_body(self._sim_agent[idx],
                                T_ref=None, 
                                include_com=True, 
                                include_p=True, 
                                include_Q=True, 
                                include_v=True, 
                                include_w=True, 
                                return_stacked=True)
    
    def state_task(self, idx):
        state = []
        
        poses, vels = [], []
        if self._ref_motion is not None:
            ref_motion = self._ref_motion[idx] 
        else:
            ref_motion = self._base_motion[idx]
        for dt in self._imit_window:
            t = np.clip(
                self.get_current_time() + dt, 
                0.0, 
                ref_motion.length())
            poses.append(ref_motion.get_pose_by_time(t))
            vels.append(ref_motion.get_velocity_by_time(t))
        state.append(self.state_imitation(self._sim_agent[idx],
                                          self._kin_agent[idx],
                                          poses,
                                          vels,
                                          include_abs=True,
                                          include_rel=True))

        return np.hstack(state)

    def state_imitation(self, 
                        sim_agent, 
                        kin_agent, 
                        poses, 
                        vels, 
                        include_abs, 
                        include_rel):

        assert len(poses) == len(vels)

        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height()))
        R_sim_inv = R_sim.transpose()
        state_sim = self._state_body(sim_agent, None, return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            state_kin = self._state_body(kin_agent, None, return_stacked=False)
            # Add pos/vel values
            if include_abs:
                state.append(np.hstack(state_kin))
            # Add difference of pos/vel values
            if include_rel:
                for j in range(len(state_sim)):
                    if len(state_sim[j])==3: 
                        state.append(state_sim[j]-state_kin[j])
                    elif len(state_sim[j])==4:
                        state.append(
                            self._pb_client.getDifferenceQuaternion(state_sim[j], state_kin[j]))
                    else:
                        raise NotImplementedError
            ''' Add facing frame differences '''
            R_kin, p_kin = conversions.T2Rp(
                kin_agent.get_facing_transform(self.get_ground_height()))
            state.append(np.dot(R_sim_inv, p_kin - p_sim))
            state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)
    
    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self.get_ground_height())
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        
        data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
        data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
        data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
        data['kin_facing_frame'] = self._kin_agent[idx].get_facing_transform(self.get_ground_height())
        data['kin_com'], data['kin_com_vel'] = self._kin_agent[idx].get_com_and_com_vel()

        return data
    
    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0
    
    def get_task_error(self, idx, data_prev, data_next, action):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        sim_facing_frame = data['sim_facing_frame']
        R_sim_f, p_sim_f = conversions.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data['sim_com'], data['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        kin_facing_frame = data['kin_facing_frame']
        R_kin_f, p_kin_f = conversions.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data['kin_com'], data['kin_com_vel']

        indices = range(len(sim_joint_p))

        if 'pose_pos' in self._reward_names[idx]:
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if 'pose_vel' in self._reward_names[idx]:
            error['pose_vel'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                error['pose_vel'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel'] /= len(indices)

        if 'ee' in self._reward_names[idx]:
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)

        if 'root' in self._reward_names[idx]:
            diff_root_p = sim_root_p - kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error['root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if 'com' in self._reward_names[idx]:
            diff_com = np.dot(R_sim_f_inv, sim_com-p_sim_f) - np.dot(R_kin_f_inv, kin_com-p_kin_f)
            diff_com_vel = sim_com_vel - kin_com_vel
            error['com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

        return error
    
    def inspect_end_of_episode_task(self):
        eoe_reason = []
        for i in range(self._num_agent):
            check = self.get_current_time() >= self._ref_motion[i].length()
            if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())
        return eoe_reason
    
    def inspect_end_of_episode_per_agent(self, idx):
        eoe_reason = super().inspect_end_of_episode_per_agent(idx)
        return eoe_reason

    def get_ground_height(self):
        return 0.0

    def get_current_time(self):
        return self._start_time + self.get_elapsed_time()

    def sample_ref_motion(self):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            idx = np.random.randint(len(self._ref_motion_all[i]))
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    import argparse
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()
    
    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, type=str)
        return parser

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset()
        def one_step(self):
            # a = np.zeros(100)
            self.env.step()
        def extra_render_callback(self):
            if self.rm.flag['follow_cam']:
                p, _, _, _ = env._sim_agent[0].get_root_state()
                self.rm.viewer.update_target_pos(p, ignore_z=True)
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_act:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Humanoid Imitation Environment=====')
    
    args = arg_parser().parse_args()

    env = Env(args.config)

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
