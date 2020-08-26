# Copyright (c) Facebook, Inc. and its affiliates.

import os

import numpy as np
import copy
from enum import Enum
from collections import deque

from fairmotion.utils import conversions
from fairmotion.processing import operations
from fairmotion.motion.motion import Pose
from fairmotion.motion.velocity import MotionWithVelocity
from fairmotion.data import bvh

import env_humanoid_tracking
import sim_agent

from abc import ABCMeta, abstractmethod

class Env(metaclass=ABCMeta):
    class ActionMode(Enum):
        Absolute=0 # Use an absolute posture as an action
        Relative=1 # Use a relative posture from a reference posture as an action
        @classmethod
        def from_string(cls, string):
            if string=="absolute": return cls.Absolute
            if string=="relative": return cls.Relative
            raise NotImplementedError
    class StateChoice(Enum):
        Body=0
        Task=1
        @classmethod
        def from_string(cls, string):
            if string=="body": return cls.Body
            if string=="task": return cls.Task
            raise NotImplementedError
    class EarlyTermChoice(Enum):
        ''' Terminate when the simulation diverges '''
        SimDiv=0
        ''' Terminate when the given time elapses '''
        SimWindow=1
        ''' Terminate when the task completes or fails '''
        TaskEnd=2
        ''' Terminate when the agents falldown '''
        Falldown=3
        ''' Terminate when the average reward goes below a specified value '''
        LowReward=4
        @classmethod
        def from_string(cls, string):
            if string=="sim_div": return cls.SimDiv
            if string=="sim_window": return cls.SimWindow
            if string=="task_end": return cls.TaskEnd
            if string=="falldown": return cls.Falldown
            if string=="low_reward": return cls.LowReward
            raise NotImplementedError
    def __init__(self, config):
        project_dir      = config['project_dir']
        char_info_module = config['character'].get('char_info_module')
        sim_char_file    = config['character'].get('sim_char_file')
        base_motion_file = config['character'].get('base_motion_file')
        ref_motion_scale = config['character'].get('ref_motion_scale')
        environment_file = config['character'].get('environment_file')
        ref_motion_file  = config['character'].get('ref_motion_file')
        self_collision   = config['character'].get('self_collision')
        actuation        = config['character'].get('actuation')
        
        ''' Append project_dir to the given file path '''
        
        if project_dir:
            for i in range(len(char_info_module)):
                char_info_module[i] = os.path.join(project_dir, char_info_module[i])
                sim_char_file[i]    = os.path.join(project_dir, sim_char_file[i])
                base_motion_file[i] = os.path.join(project_dir, base_motion_file[i])
            if environment_file is not None:
                for i in range(len(environment_file)):
                    environment_file[i] = os.path.join(project_dir, environment_file[i])

        ''' Create a base tracking environment '''

        self._base_env = env_humanoid_tracking.Env(
            fps_sim=config['fps_sim'],
            fps_act=config['fps_con'],
            verbose=config['verbose'],
            char_info_module=char_info_module,
            sim_char_file=sim_char_file,
            ref_motion_scale=ref_motion_scale,
            self_collision=self_collision,
            contactable_body=config['early_term'].get('falldown_contactable_body'),
            actuation=actuation,
            )

        self._pb_client = self._base_env._pb_client
        self._dt_con = 1.0/config['fps_con']

        ''' Copy some of frequently used attributes from the base environemnt '''
        self._num_agent = self._base_env._num_agent
        assert self._num_agent == len(base_motion_file)
        self._sim_agent = [self._base_env._agent[i] for i in range(self._num_agent)]
        self._v_up = self._base_env._v_up

        ''' State '''
        self._state_choices = [Env.StateChoice.from_string(s) for s in config['state']['choices']]

        ''' Early Terminations '''
        self._early_term_choices = [Env.EarlyTermChoice.from_string(s) for s in config['early_term']['choices']]

        self._reward_fn_def = config['reward']['fn_def']
        self._reward_fn_map = config['reward']['fn_map']
        self._reward_names = [self.get_reward_names(
            self._reward_fn_def[self._reward_fn_map[i]]) for i in range(self._num_agent)]

        '''
        Check the existence of reward definitions, which are defined in our reward map
        '''
        assert len(self._reward_fn_map) == self._num_agent
        for key in self._reward_fn_map:
            assert key in self._reward_fn_def.keys()

        self._verbose = config['verbose']

        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            self._et_low_reward_thres = config['early_term']['low_reward_thres']
            self._rew_queue = self._num_agent * [None]
            for i in range(self._num_agent):
                self._rew_queue[i] = deque(maxlen=int(1.0/self._dt_con))
        
        ''' The environment automatically terminates after 'sim_window' seconds '''
        if Env.EarlyTermChoice.SimWindow in self._early_term_choices:
            self._sim_window_time = config['early_term']['sim_window_time']
        ''' 
        The environment continues for "eoe_margin" seconds after end-of-episode is set by TRUE.
        This is useful for making the controller work for boundaries of reference motions
        '''
        self._eoe_margin = config['early_term']['eoe_margin']

        self._action_type = Env.ActionMode.from_string(config['action']['type'])

        ''' Base motion defines the initial posture (like t-pose) '''

        self._base_motion = []
        for i in range(self._num_agent):
            m = bvh.load(file=base_motion_file[i],
                         motion=MotionWithVelocity(),
                         scale=1.0, 
                         load_skel=True,
                         load_motion=True,
                         v_up_skel=self._sim_agent[i]._char_info.v_up, 
                         v_face_skel=self._sim_agent[i]._char_info.v_face, 
                         v_up_env=self._sim_agent[i]._char_info.v_up_env)
            m = MotionWithVelocity.from_motion(m)
            self._base_motion.append(m)

        ''' Create Kinematic Agents '''
        
        self._kin_agent = []
        for i in range(self._num_agent):
            self._kin_agent.append(
                sim_agent.SimAgent(pybullet_client=self._base_env._pb_client, 
                                   model_file=sim_char_file[i],
                                   char_info=self._sim_agent[i]._char_info,
                                   ref_scale=ref_motion_scale[i],
                                   self_collision=self_collision[i],
                                   kinematic_only=True,
                                   verbose=config['verbose']))

        ''' 
        Define the action space of this environment.
        Here I used a 'normalizer' where 'real' values correspond to joint angles,
        and 'norm' values correspond to the output value of NN policy.
        The reason why it is used is that NN policy somtimes could output values that
        are within much larger or narrow range than we need for the environment.
        For example, if we apply tanh activation function at the last layer of NN,
        the output are always within (-1, 1), but we need bigger values for joint angles 
        because 1 corresponds only to 57.3 degree.
        '''

        self._action_space = []
        for i in range(self._num_agent):
            dim = self._sim_agent[i].get_num_dofs()
            normalizer = operations.Normalizer(
                real_val_max=config['action']['range_max']*np.ones(dim),
                real_val_min=config['action']['range_min']*np.ones(dim),
                norm_val_max=config['action']['range_max_pol']*np.ones(dim),
                norm_val_min=config['action']['range_min_pol']*np.ones(dim),
                apply_clamp=True)
            self._action_space.append(normalizer)

        self._com_vel = self._num_agent * [None]
        for i in range(self._num_agent):
            self._com_vel[i] = deque(maxlen=int(1.0/self._dt_con))

        ''' 
        Any necessary information needed for training this environment.
        This can be set by calling "set_learning_info". 
        '''
        self._learning_info = {}

        self.add_noise = config['add_noise']

    def action_range(self, idx):
        return self._action_space[idx].real_val_min, self._action_space[idx].real_val_max

    def dim_action(self, idx):
        return self._action_space[idx].dim

    def dim_state(self, idx):
        return len(self.state(idx))

    def dim_state_body(self, idx):
        return len(self.state_body(idx))

    def dim_state_task(self, idx):
        return len(self.state_task(idx))

    def set_learning_info(self, info):
        self._learning_info = info

    def update_learning_info(self, info):
        self._learning_info.update(info)

    def agent_avg_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([(agent.get_root_state())[0] for agent in agents], axis=0)

    def agent_ave_facing_position(self, agents=None):
        if agents is None: agents=self._sim_agent
        return np.mean([agent.get_facing_position(self.get_ground_height()) for agent in agents], axis=0)

    def throw_obstacle(self):
        size = np.random.uniform(0.1, 0.3, 3)
        p = self.agent_avg_position()
        self._base_env.throw_obstacle(size, p)

    def split_action(self, action):
        assert len(action)%self._num_agent == 0
        dim_action = len(action)//self._num_agent
        actions = []
        idx = 0
        for i in range(self._num_agent):
            actions.append(action[idx:idx+dim_action])
            idx += dim_action
        return actions

    def compute_target_pose(self, idx, action):
        agent = self._sim_agent[idx]
        char_info = agent._char_info
        
        ''' the current posture should be deepcopied because action will modify it '''
        if self._action_type == Env.ActionMode.Relative:
            ref_pose = copy.deepcopy(self.get_current_pose_from_motion(idx))
        else:
            ref_pose = copy.deepcopy(self._base_motion[idx].get_pose_by_frame(0))

        a_real = self._action_space[idx].norm_to_real(action)

        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Fixed joint will not be affected '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' If the joint do not have correspondance, use the reference posture itself'''
            if char_info.bvh_map[j] == None:
                continue
            if self._action_type == Env.ActionMode.Relative:
                T = ref_pose.get_transform(char_info.bvh_map[j], local=True)
            elif self._action_type == Env.ActionMode.Absolute:
                T = ref_pose.skel.get_joint(char_info.bvh_map[j]).xform_from_parent_joint
            else:
                raise NotImplementedError
            R, p = conversions.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = conversions.A2R(a_real[dof_cnt:dof_cnt+3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                axis = agent.get_joint_axis(j)
                angle = a_real[dof_cnt:dof_cnt+1]
                dR = conversions.A2R(axis*angle)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = conversions.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)

        return ref_pose

    def compute_init_pose_vel(self, add_noise):
        '''
        This compute initial poses and velocities for all agents.
        The returned poses and velocites will be the initial pose and
        velocities of the simulated agent.
        '''
        init_poses, init_vels = [], []
        for i in range(self._num_agent):
            cur_pose = self._base_motion[i].get_pose_by_frame(0)
            cur_vel = self._base_motion[i].get_velocity_by_frame(0)
            if add_noise:
                cur_pose, cur_vel = self._base_env.add_noise_to_pose_vel(
                    self._sim_agent[i], cur_pose, cur_vel)
            init_poses.append(cur_pose)
            init_vels.append(cur_vel)
        return init_poses, init_vels

    def callback_reset_prev(self, info):
        '''
        This is called right before the main reset fn. is called.
        '''
        return

    def callback_reset_after(self, info):
        '''
        This is called right after the main reset fn. is called.
        '''
        return
    
    def reset(self, info):
        
        self.callback_reset_prev(info)

        self._target_pose = [None for i in range(self._num_agent)]
        self._init_poses, self._init_vels = self.compute_init_pose_vel(info)

        self._base_env.reset(time=0.0,
                             poses=self._init_poses, 
                             vels=self._init_vels)
        
        self._end_of_episode = False
        self._end_of_episode_reason = []

        self._end_of_episode_intermediate = False
        self._end_of_episode_reason_intermediate = []
        self._time_elapsed_after_end_of_episode = 0.0

        for i in range(self._num_agent):
            self._com_vel[i].clear()
            self._com_vel[i].append(self._sim_agent[i].get_com_and_com_vel()[1])

        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            for i in range(self._num_agent):
                self._rew_queue[i].clear()
                for j in range(self._rew_queue[i].maxlen):
                    self._rew_queue[i].append(self.reward_max())

        self.callback_reset_after(info)

    def callback_step_prev(self):
        return

    def callback_step_after(self):
        return

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')
    
    def step(self, action):

        self.callback_step_prev()

        ''' Collect data for reward computation before the current step'''
        rew_data_prev = [self.reward_data(i) for i in range(self._num_agent)]

        assert len(action) == self._num_agent
        
        for i in range(self._num_agent):
            if isinstance(action[i], Pose):
                self._target_pose[i] = action[i]
            elif isinstance(action[i], np.ndarray):
                self._target_pose[i] = self.compute_target_pose(i, action[i])
            else:
                print(type(action[i]))
                raise NotImplementedError
        
        for i in range(self._num_agent):
            self._com_vel[i].append(self._sim_agent[i].get_com_and_com_vel()[1])
        
        ''' Update simulation '''
        self._base_env.step(self._target_pose)

        self.callback_step_after()

        ''' Collect data for reward computation after the current step'''
        rew_data_next = [self.reward_data(i) for i in range(self._num_agent)]

        ''' 
        Check conditions for end-of-episode. 
        If 'eoe_margin' is larger than zero, the environment will continue for some time.
        '''
        
        if not self._end_of_episode_intermediate:
            eoe_reason = []
            for i in range(self._num_agent):
                eoe_reason += self.inspect_end_of_episode_per_agent(i)
            if Env.EarlyTermChoice.TaskEnd in self._early_term_choices:
                eoe_reason += self.inspect_end_of_episode_task()

            self._end_of_episode_intermediate = len(eoe_reason) > 0
            self._end_of_episode_reason_intermediate = eoe_reason

        if self._end_of_episode_intermediate:
            self._time_elapsed_after_end_of_episode += self._dt_con
            if self._time_elapsed_after_end_of_episode >= self._eoe_margin:
                self._end_of_episode = True
                self._end_of_episode_reason = self._end_of_episode_reason_intermediate

        ''' Compute rewards '''
        
        rews, infos = [], []
        for i in range(self._num_agent):
            r, rd = self.reward(i, rew_data_prev, rew_data_prev, action)
            rews.append(r)
            info = {
                'eoe_reason': self._end_of_episode_reason,
                'rew_info': rd,
                'learning_info': self._learning_info
            }
            infos.append(info)
            if Env.EarlyTermChoice.LowReward in self._early_term_choices:
                self._rew_queue[i].append(r)

        self.print_log_in_step()
        
        return rews, infos

    def state(self, idx):
        state = []
        
        if Env.StateChoice.Body in self._state_choices:
            state.append(self.state_body(idx))
        if Env.StateChoice.Task in self._state_choices:
            state.append(self.state_task(idx))

        return np.hstack(state)

    @abstractmethod
    def state_body(self, idx):
        '''
        This returns proprioceptive state of an agent as a numpy array
        '''
        raise NotImplementedError

    def _state_body(self, 
                    agent, 
                    T_ref=None, 
                    include_com=True, 
                    include_p=True, 
                    include_Q=True, 
                    include_v=True, 
                    include_w=True, 
                    return_stacked=True):
        if T_ref is None: 
            T_ref = agent.get_facing_transform(self.get_ground_height())

        R_ref, p_ref = conversions.T2Rp(T_ref)
        R_ref_inv = R_ref.transpose()

        link_states = []
        link_states.append(agent.get_root_state())
        ps, Qs, vs, ws = agent.get_link_states()
        for j in agent._joint_indices:
            link_states.append((ps[j], Qs[j], vs[j], ws[j]))

        state = []
        for i, s in enumerate(link_states):
            p, Q, v, w = s[0], s[1], s[2], s[3]
            if include_p:
                p_rel = np.dot(R_ref_inv, p - p_ref)
                state.append(p_rel) # relative position w.r.t. the reference frame
            if include_Q:
                Q_rel = conversions.R2Q(np.dot(R_ref_inv, conversions.Q2R(Q)))
                Q_rel = operations.Q_op(Q_rel, op=["normalize", "halfspace"])
                state.append(Q_rel) # relative rotation w.r.t. the reference frame
            if include_v:
                v_rel = np.dot(R_ref_inv, v)
                state.append(v_rel) # relative linear vel w.r.t. the reference frame
            if include_w:
                w_rel = np.dot(R_ref_inv, w)
                state.append(w_rel) # relative angular vel w.r.t. the reference frame
            if include_com:
                if i==0:
                    p_com = agent._link_masses[i] * p
                    v_com = agent._link_masses[i] * v
                else:
                    p_com += agent._link_masses[i] * p
                    v_com += agent._link_masses[i] * v

        if include_com:
            p_com /= agent._link_total_mass
            v_com /= agent._link_total_mass
            state.append(np.dot(R_ref_inv, p_com - p_ref))
            state.append(np.dot(R_ref_inv, v_com))
        
        if return_stacked:
            return np.hstack(state)
        else:
            return state

    @abstractmethod
    def state_task(self, idx):
        '''
        This returns a task-specifit state (numpy array)
        '''     
        raise NotImplementedError

    @abstractmethod
    def reward_data(self, idx):
        '''
        This returns a dictionary that includes data to compute reward value
        '''
        raise NotImplementedError

    @abstractmethod
    def reward_max(self):
        '''
        This returns a maximum reward value
        '''
        raise NotImplementedError

    @abstractmethod
    def reward_min(self):
        '''
        This returns a minimum reward value
        '''
        raise NotImplementedError

    def return_max(self, gamma):
        '''
        This returns a maximum return (sum of rewards)
        '''
        assert gamma < 1.0
        return self.reward_max() / (1.0 - gamma)

    def return_min(self, gamma):
        '''
        This returns a minimum return (sum of rewards)
        '''
        assert gamma < 1.0
        return self.reward_min() / (1.0 - gamma)

    @abstractmethod
    def get_task_error(self, idx, data_prev, data_next, action):
        '''
        This computes a task-specific error and 
        returns a dictionary that includes those errors
        '''
        raise NotImplementedError

    def reward(self, idx, data_prev, data_next, action):
        '''
        This returns a reward, and a dictionary
        '''   
        
        error = self.get_task_error(idx, data_prev, data_next, action)

        rew_fn_def = self._reward_fn_def[self._reward_fn_map[idx]]
        rew, rew_info = self.compute_reward(error, rew_fn_def)

        return rew, rew_info

    def get_reward_names(self, fn_def):
        rew_names = set()
        op = fn_def['op']

        if op in ['add', 'mul']:
            for child in fn_def['child_nodes']:
                rew_names = rew_names.union(self.get_reward_names(child))
        elif op == 'leaf':
            rew_names.add(fn_def['name'])
        else:
            raise NotImplementedError

        return rew_names

    def pretty_print_rew_info(self, rew_info, prefix=str()):
        print("%s > name:   %s"%(prefix, rew_info['name']))
        print("%s   value:  %s"%(prefix, rew_info['value']))
        print("%s   weight: %s"%(prefix, rew_info['weight']))
        print("%s   op: %s"%(prefix, rew_info['op']))
        for child in rew_info["child_nodes"]:
            self.pretty_print_rew_info(child, prefix+"\t")

    def compute_reward(self, error, fn_def):
        ''' 
        This computes a reward by using 
        task-specific errors and the reward definition tree
        '''
        op = fn_def['op']
        n = fn_def['name'] if 'name' in fn_def.keys() else 'noname'
        w = fn_def['weight'] if 'weight' in fn_def.keys() else 1.0

        rew_info = {'name': n, 'value': 0.0, 'op': op, 'weight': w, 'child_nodes': []}

        if op in ['add', 'sum']:
            rew = 0.0
            for child in fn_def['child_nodes']:
                r, rd = self.compute_reward(error, child)
                rew += r
                rew_info['child_nodes'].append(rd)
        elif op in ['mul', 'multiply']:
            rew = 1.0
            for child in fn_def['child_nodes']:
                r, rd = self.compute_reward(error, child)
                rew *= r
                rew_info['child_nodes'].append(rd)
        elif op == 'leaf':
            if 'kernel' in fn_def.keys():
                kernel = fn_def['kernel']
            else:
                kernel = None

            if 'weight_schedule' in fn_def.keys():
                timesteps_total = self._learning_info['timesteps_total']
                w *= operations.lerp_from_paired_list(
                    timesteps_total, fn_def['weight_schedule'])
            
            if kernel is None or kernel['type'] == "none":
                e = error[n]
            elif kernel['type'] == "gaussian":
                e = np.exp(-kernel['scale']*error[n])
            else:
                raise NotImplementedError
            
            rew = w*e
        else:
            raise NotImplementedError

        rew_info['value'] = rew

        return rew, rew_info

    @abstractmethod
    def inspect_end_of_episode_task(self):
        '''
        This checks whether task-specific END-OF-EPISODE events happen and
        returns a list that includes reasons
        '''
        raise NotImplementedError

    def inspect_end_of_episode_per_agent(self, idx):
        eoe_reason = []
        name = self._sim_agent[idx].get_name()

        if Env.EarlyTermChoice.Falldown in self._early_term_choices:
            check = self._base_env.check_falldown(self._sim_agent[idx])
            if check: eoe_reason.append('[%s] falldown'%name)
        if Env.EarlyTermChoice.SimDiv in self._early_term_choices:
            check = self._base_env.is_sim_div(self._sim_agent[idx])
            if check: eoe_reason.append('[%s] sim_div'%name)
        if Env.EarlyTermChoice.SimWindow in self._early_term_choices:
            check = self.get_elapsed_time() > self._sim_window_time
            if check: eoe_reason.append('[%s] sim_window'%name)
        if Env.EarlyTermChoice.LowReward in self._early_term_choices:
            check = np.mean(list(self._rew_queue[idx])) < self._et_low_reward_thres * self.reward_max()
            if check: eoe_reason.append('[%s] low_rewards'%name)

        return eoe_reason

    @abstractmethod
    def get_ground_height(self):
        '''
        This returns height of the ground
        '''
        raise NotImplementedError

    def get_elapsed_time(self):
        '''
        This returns the elpased time after the environment was reset
        '''
        return self._base_env._elapsed_time

    def set_elapsed_time(self, time):
        self._base_env._elapsed_time = time

    def render(self, rm):
        colors = rm.COLORS_FOR_AGENTS

        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        self._base_env.render(rm,
                              ground_height=self.get_ground_height())

        if rm.flag['target_pose']:
            for i in range(self._num_agent):
                if self._target_pose[i] is None: continue
                agent = self._kin_agent[i]
                agent_state = agent.save_states()
                agent.set_pose(self._target_pose[i])
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[colors[i][0], colors[i][1], colors[i][2], 0.5])
                rm.gl.glPopAttrib()
                agent.restore_states(agent_state)

        if rm.flag['kin_model']:
            for i in range(self._num_agent):
                agent = self._kin_agent[i]                
                rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                rm.bullet_render.render_model(self._pb_client, 
                                              agent._body_id,
                                              draw_link=True,
                                              draw_link_info=False,
                                              draw_joint=rm.flag['joint'],
                                              draw_joint_geom=False, 
                                              ee_indices=agent._char_info.end_effector_indices,
                                              color=[colors[i][0], colors[i][1], colors[i][2], 0.5])
                if rm.flag['com_vel']:
                    p, Q, v, w = agent.get_root_state()
                    p, v = agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
                rm.gl.glPopAttrib()
