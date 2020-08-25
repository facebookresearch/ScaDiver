# Copyright (c) Facebook, Inc. and its affiliates.

import os

''' 
This forces the environment to use only 1 cpu when running.
This is helpful to launch multiple environment simulatenously.
'''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import copy

import pybullet as pb
import pybullet_data

from bullet import bullet_client
from bullet import bullet_utils as bu

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.utils import constants

import sim_agent
import sim_obstacle
import importlib.util

class Env(object):
    '''
    This environment defines a base environment where the simulated 
    characters exist and they are controlled by tracking controllers
    '''
    def __init__(self, 
                 fps_sim, 
                 fps_act, 
                 char_info_module,
                 sim_char_file,
                 ref_motion_scale,
                 actuation,
                 self_collision=None,
                 contactable_body=None,
                 verbose=False,
                 ):
        self._num_agent = len(sim_char_file)
        assert self._num_agent > 0
        assert self._num_agent == len(char_info_module)
        assert self._num_agent == len(ref_motion_scale)

        self._char_info = []
        for i in range(self._num_agent):
            ''' Load Character Info Moudle '''
            spec = importlib.util.spec_from_file_location(
                "char_info%d"%(i), char_info_module[i])
            char_info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(char_info)
            self._char_info.append(char_info)

            ''' Modfiy Contactable Body Parts '''
            if contactable_body:
                contact_allow_all = True if 'all' in contactable_body else False
                for joint in list(char_info.contact_allow_map.keys()):
                    char_info.contact_allow_map[joint] = \
                        contact_allow_all or char_info.joint_name[joint] in contactable_body

        self._v_up = self._char_info[0].v_up_env

        ''' Define PyBullet Client '''
        self._pb_client = bullet_client.BulletClient(
            connection_mode=pb.DIRECT, options=' --opengl2')
        self._pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        ''' timestep for physics simulation '''
        self._dt_sim = 1.0/fps_sim
        ''' timestep for control of dynamic controller '''
        self._dt_act = 1.0/fps_act

        if fps_sim%fps_act != 0:
            raise Exception('FPS_SIM should be a multiples of FPS_ACT')
        self._num_substep = fps_sim//fps_act

        self._verbose = verbose

        self.setup_physics_scene(sim_char_file, 
                                 self._char_info, 
                                 ref_motion_scale, 
                                 self_collision,
                                 actuation)

        ''' Elapsed time after the environment starts '''
        self._elapsed_time = 0.0
        ''' For tracking the length of current episode '''
        self._episode_len = 0.0
        
        ''' Create a Manager for Handling Obstacles '''
        self._obs_manager = sim_obstacle.ObstacleManager(
            self._pb_client, self._dt_act, self._char_info[0].v_up_env)

        ''' Save the initial pybullet state to clear all thing before calling reset '''
        self._init_state = None
        self.reset()
        self._init_state = self._pb_client.saveState()

    def setup_physics_scene(self, sim_char_file, char_info, ref_motion_scale, self_collision, actuation):
        self._pb_client.resetSimulation()
        
        self.create_ground()
        
        self._agent = []
        for i in range(self._num_agent):
            self._agent.append(sim_agent.SimAgent(name='sim_agent_%d'%(i),
                                                  pybullet_client=self._pb_client, 
                                                  model_file=sim_char_file[i], 
                                                  char_info=char_info[i], 
                                                  ref_scale=ref_motion_scale[i],
                                                  self_collision=self_collision[i],
                                                  actuation=actuation[i],
                                                  kinematic_only=False,
                                                  verbose=self._verbose))

    def create_ground(self):
        ''' Create Plane '''
        if np.allclose(np.array([0.0, 0.0, 1.0]), self._v_up):
            R_plane = constants.eye_R()
        else:
            R_plane = math.R_from_vectors(np.array([0.0, 0.0, 1.0]), self._v_up)
        self._plane_id = \
            self._pb_client.loadURDF(
                "plane_implicit.urdf", 
                [0, 0, 0], 
                conversions.R2Q(R_plane), 
                useMaximalCoordinates=True)
        self._pb_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)

        ''' Dynamics parameters '''
        assert np.allclose(np.linalg.norm(self._v_up), 1.0)
        gravity = -9.8 * self._v_up
        self._pb_client.setGravity(gravity[0], gravity[1], gravity[2])
        self._pb_client.setTimeStep(self._dt_sim)
        self._pb_client.setPhysicsEngineParameter(numSubSteps=2)
        self._pb_client.setPhysicsEngineParameter(numSolverIterations=10)
        # self._pb_client.setPhysicsEngineParameter(solverResidualThreshold=1e-10)

    def check_collision(self, body_id1, body_id2, link_id1=None, link_id2=None):
        ''' collision between two bodies '''
        pts = self._pb_client.getContactPoints(
            bodyA=body_id1, bodyB=body_id2, linkIndexA=link_id1, linkIndexB=link_id2)
        return len(p) > 0

    # def check_falldown(self, agent, plane_id=None):
    #     ''' check if any non-allowed body part hits the ground '''
    #     if plane_id is None: plane_id = self._plane_id
    #     pts = self._pb_client.getContactPoints()
    #     for p in pts:
    #         part = None
    #         #ignore self-collision
    #         if p[1] == p[2]: continue
    #         if p[1] == agent._body_id and p[2] == plane_id: part = p[3]
    #         if p[2] == agent._body_id and p[1] == plane_id: part = p[4]
    #         #ignore collision of other agents
    #         if part == None: continue
    #         if not agent._char_info.contact_allow_map[part]: return True
    #     return False

    def check_falldown(self, agent, plane_id=None):
        ''' check if any non-allowed body part hits the ground '''
        if plane_id is None: plane_id = self._plane_id
        pts = self._pb_client.getContactPoints(
            bodyA=agent._body_id, bodyB=plane_id)
        for p in pts:
            part = p[3] if p[1] == agent._body_id else p[4]
            if agent._char_info.contact_allow_map[part]:
                continue
            else:
                return True
        return False

    def is_sim_div(self, agent):
        ''' TODO: check divergence of simulation '''
        return False

    def step(self, target_poses=[]):
        ''' 
        One Step-forward Simulation 
        '''

        ''' Increase elapsed time '''
        self._elapsed_time += self._dt_act
        self._episode_len += self._dt_act

        ''' Update simulation '''
        for _ in range(self._num_substep):
            for i, target_pose in enumerate(target_poses):
                self._agent[i].actuate(pose=target_pose,
                                       vel=None)
            self._pb_client.stepSimulation()

        self._obs_manager.update()

    def reset(self, time=0.0, poses=None, vels=None, pb_state_id=None):

        ''' remove obstacles in the scene '''
        self._obs_manager.clear()

        ''' 
        Restore internal pybullet state 
        by uisng the saved info when Env was initially created  
        '''
        if pb_state_id is not None:
            self._pb_client.restoreState(pb_state_id)

        self._elapsed_time = time

        if poses is None:
            if self._init_state is not None:
                self._pb_client.restoreState(self._init_state)
        else:
            for i in range(self._num_agent):
                pose = poses[i]
                vel = None if vels is None else vels[i]
                self._agent[i].set_pose(pose, vel)

        self._episode_len = 0.0

    def add_noise_to_pose_vel(self, agent, pose, vel=None, return_as_copied=True):
        '''
        Add a little bit of noise to the given pose and velocity
        '''

        ref_pose = copy.deepcopy(pose) if return_as_copied else pose
        if vel:
            ref_vel = copy.deepcopy(vel) if return_as_copied else vel
        dof_cnt = 0
        for j in agent._joint_indices:
            joint_type = agent.get_joint_type(j)
            ''' Ignore fixed joints '''
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            ''' Ignore if there is no corresponding joint '''
            if agent._char_info.bvh_map[j] == None:
                continue
            T = ref_pose.get_transform(agent._char_info.bvh_map[j], local=True)
            R, p = conversions.T2Rp(T)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                dR = math.random_rotation(
                    mu_theta=agent._char_info.noise_pose[j][0],
                    sigma_theta=agent._char_info.noise_pose[j][1],
                    lower_theta=agent._char_info.noise_pose[j][2],
                    upper_theta=agent._char_info.noise_pose[j][3])
                dof_cnt += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                theta = math.truncnorm(
                    mu=agent._char_info.noise_pose[j][0],
                    sigma=agent._char_info.noise_pose[j][1],
                    lower=agent._char_info.noise_pose[j][2],
                    upper=agent._char_info.noise_pose[j][3])
                joint_axis = agent.get_joint_axis(j)
                dR = conversions.A2R(joint_axis*theta)
                dof_cnt += 1
            else:
                raise NotImplementedError
            T_new = conversions.Rp2T(np.dot(R, dR), p)
            ref_pose.set_transform(agent._char_info.bvh_map[j], T_new, do_ortho_norm=False, local=True)
            if vel is not None:
                dw = math.truncnorm(
                    mu=np.full(3, agent._char_info.noise_vel[j][0]),
                    sigma=np.full(3, agent._char_info.noise_vel[j][1]),
                    lower=np.full(3, agent._char_info.noise_vel[j][2]),
                    upper=np.full(3, agent._char_info.noise_vel[j][3]))
                ref_vel.data_local[j][:3] += dw
        return ref_pose, ref_vel

    def render(self, rm, ground_height=0.0):
        colors = rm.COLORS_FOR_AGENTS

        rm.gl.glEnable(rm.gl.GL_LIGHTING)
        rm.gl.glEnable(rm.gl.GL_BLEND)
        rm.gl.glBlendFunc(rm.gl.GL_SRC_ALPHA, rm.gl.GL_ONE_MINUS_SRC_ALPHA)

        for i in range(self._num_agent):
            sim_agent = self._agent[i]
            char_info = self._char_info[i]
            if rm.flag['sim_model']:
                rm.gl.glEnable(rm.gl.GL_DEPTH_TEST)
                if rm.flag['shadow']:
                    rm.gl.glPushMatrix()
                    d = np.array([1, 1, 1])
                    d = d - math.projectionOnVector(d, char_info.v_up_env)
                    offset = (0.001 + ground_height) * char_info.v_up_env
                    rm.gl.glTranslatef(offset[0], offset[1], offset[2])
                    rm.gl.glScalef(d[0], d[1], d[2])
                    rm.bullet_render.render_model(self._pb_client, 
                                               sim_agent._body_id, 
                                               draw_link=True, 
                                               draw_link_info=False, 
                                               draw_joint=False, 
                                               draw_joint_geom=False, 
                                               ee_indices=None, 
                                               color=[0.5,0.5,0.5,1.0],
                                               lighting=False)
                    rm.gl.glPopMatrix()

                rm.bullet_render.render_model(self._pb_client, 
                                              sim_agent._body_id,
                                              draw_link=True, 
                                              draw_link_info=True, 
                                              draw_joint=rm.flag['joint'], 
                                              draw_joint_geom=True, 
                                              ee_indices=char_info.end_effector_indices, 
                                              color=colors[i])
                if rm.flag['collision'] and self._elapsed_time > 0.0:
                    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                    rm.gl.glEnable(rm.gl.GL_BLEND)
                    rm.bullet_render.render_contacts(self._pb_client, sim_agent._body_id)
                    rm.gl.glPopAttrib()
                if rm.flag['com_vel']:
                    p, Q, v, w = sim_agent.get_root_state()
                    p, v = sim_agent.get_com_and_com_vel()
                    rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0, 0, 0, 1])
                if rm.flag['facing_frame']:
                    rm.gl.glPushAttrib(rm.gl.GL_LIGHTING|rm.gl.GL_DEPTH_TEST|rm.gl.GL_BLEND)
                    rm.gl.glEnable(rm.gl.GL_BLEND)
                    rm.gl_render.render_transform(
                        sim_agent.get_facing_transform(ground_height), 
                        scale=0.5, 
                        use_arrow=True)
                    rm.gl.glPopAttrib()

        if rm.flag['obstacle']:
            self._obs_manager.render()

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()

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
    
    print('=====Motion Tracking Controller=====')

    env = Env(fps_sim=480,
              fps_act=30, 
              verbose=False,
              char_info_module=['amass_char_info.py'],
              sim_char_file=['data/character/amass.urdf'],
              ref_motion_scale=[1.0],
              self_collision=[True],
              actuation=["spd"])

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
