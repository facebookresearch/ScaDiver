import copy
import numpy as np
import argparse
import random

import gym
from gym.spaces import Box

import env_humanoid_imitation as my_env
import env_renderer as er
import render_module as rm

import os

class HumanoidImitation(gym.Env):
    def __init__(self, env_config):
        self.base_env = my_env.Env(env_config)
        assert self.base_env._num_agent == 1
        
        ob_scale = 1000.0
        dim_state = self.base_env.dim_state(0)
        dim_state_body = self.base_env.dim_state_body(0)
        dim_state_task = self.base_env.dim_state_task(0)
        dim_action = self.base_env.dim_action(0)
        action_range_min, action_range_max = self.base_env.action_range(0)
        self.observation_space = \
            Box(-ob_scale * np.ones(dim_state),
                ob_scale * np.ones(dim_state),
                dtype=np.float64)
        self.observation_space_body = \
            Box(-ob_scale * np.ones(dim_state_body),
                ob_scale * np.ones(dim_state_body),
                dtype=np.float64)
        self.observation_space_task = \
            Box(-ob_scale * np.ones(dim_state_task),
                ob_scale * np.ones(dim_state_task),
                dtype=np.float64)
        self.action_space = \
            Box(action_range_min,
                action_range_max,
                dtype=np.float64)

    def state(self):
        return self.base_env.state(idx=0)

    def reset(self, start_time=None, add_noise=None):
        if not self.base_env._initialized:
            self.base_env.create()
        self.base_env.reset({
            'start_time': start_time, 
            'add_noise': add_noise,
            })
        return self.base_env.state(idx=0)

    def step(self, action):
        rew, info = self.base_env.step([action])
        obs = self.state()
        eoe = self.base_env._end_of_episode
        return obs, rew[0], eoe, info[0]

class EnvRenderer(er.EnvRenderer):
    def __init__(self, trainer, **kwargs):
        from fairmotion.viz.utils import TimeChecker
        super().__init__(**kwargs)
        self.trainer = trainer
        self.time_checker_auto_play = TimeChecker()
        self.explore = False
    def one_step(self):
        s1 = self.env.state()
        a = self.trainer.compute_action(s1, explore=self.explore)
        s2, rew, eoe, info = self.env.step(a)
    def extra_render_callback(self):
        if self.rm.flag['follow_cam']:
            p, _, _, _ = self.env.base_env._sim_agent[0].get_root_state()
            self.update_target_pos(p, ignore_z=True)
        self.env.base_env.render(self.rm)
    def extra_overlay_callback(self):
        model = self.trainer.get_policy().model
        if hasattr(model, 'gate_function'):
            expert_weights = model.gate_function()
            num_experts = model.num_experts()
            w, h = self.window_size
            w_bar, h_bar = 150, 20
            origin = np.array([0.95*w-w_bar, 0.95*h-h_bar])
            pos = origin.copy()
            for i in reversed(range(num_experts)):
                self.rm.gl_render.render_text(
                    "Expert%d"%(i), 
                    pos=pos-np.array([75, -0.8*h_bar]), 
                    font=self.rm.glut.GLUT_BITMAP_9_BY_15)
                w_i = expert_weights[0][i] if expert_weights is not None else 0.0
                self.rm.gl_render.render_progress_bar_2D_horizontal(
                    w_i, origin=pos, width=w_bar, 
                    height=h_bar, color_input=self.rm.COLORS_FOR_EXPERTS[i])
                pos += np.array([0.0, -h_bar])                
    def extra_idle_callback(self):
        time_elapsed = self.time_checker_auto_play.get_time(restart=False)
        if self.rm.flag['auto_play'] and time_elapsed >= self.env.base_env._dt_con:
            self.time_checker_auto_play.begin()
            self.one_step()
    def extra_keyboard_callback(self, key):
        if key == b'r':
            s = self.env.reset()
        elif key == b'R':
            s = self.env.reset(start_time=0.0)
        elif key == b' ':
            self.time_checker_auto_play.begin()
            self.one_step()
        elif key == b'e':
            self.explore = not self.explore
            print('Exploration:', self.explore)
        elif key == b's':
            model = self.trainer.get_policy().model
            model.save_weights_body_encoder('data/temp/body_encoder.pt')
            model.save_weights_motor_decoder('data/temp/motor_decoder.pt')
        elif key == b'c':
            ''' Read a directory for saving images and try to create it '''
            subdir = input("Enter subdirectory for screenshot file: ")
            dir = os.path.join("data/screenshot/", subdir)
            try:
                os.makedirs(dir, exist_ok = True)
            except OSError:
                print("Invalid Subdirectory")
                return
            for i in range(1):
                try:
                    os.makedirs(dir, exist_ok = True)
                except OSError:
                    print("Invalid Subdirectory")
                    continue
                cnt_screenshot = 0
                while True:
                    name = 'screenshot_%04d'%(cnt_screenshot)
                    self.one_step()
                    self.render()
                    self.save_screen(dir=dir, name=name)
                    print('\rsave_screen(%4.4f) / %s' % \
                        (self.env.base_env.get_elapsed_time(), os.path.join(dir,name)), end=" ")
                    cnt_screenshot += 1
                    if self.env.base_env._end_of_episode:
                        break
                print("\n")

def default_cam():
    return rm.camera.Camera(pos=np.array([0.0, 3.0, 2.0]),
                            origin=np.array([0.0, 0.0, 0.0]), 
                            vup=np.array([0.0, 0.0, 1.0]), 
                            fov=60.0)

env_cls = HumanoidImitation

def config_override(spec):
    env = env_cls(spec["config"]["env_config"])

    model_config = copy.deepcopy(spec["config"]["model"])
    model = model_config.get("custom_model")
    if model and model == "task_agnostic_policy_type1":
        model_config.get("custom_options").update({
            "observation_space_body": copy.deepcopy(env.observation_space_body),
            "observation_space_task": copy.deepcopy(env.observation_space_task),
        })

    del env

    config = {
        # "callbacks": {},
        "model": model_config,
    }
    return config
