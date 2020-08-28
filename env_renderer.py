# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import render_module as rm
from fairmotion.viz import glut_viewer

rm.initialize()

def axis_to_str(axis):
    if np.allclose(axis, np.array([1.0, 0.0, 0.0])):
        return 'x'
    elif np.allclose(axis, np.array([0.0, 1.0, 0.0])):
        return 'y'
    elif np.allclose(axis, np.array([0.0, 0.0, 1.0])):
        return 'z'
    else: 
        raise Exception

class EnvRenderer(glut_viewer.Viewer):
    def __init__(self,
                 env=None,
                 title="env_renderer",
                 cam=None,
                 size=(1200, 900)):
        super().__init__(title, cam, size)
        self.rm = rm
        self.env = env
    def save_screen(self, dir, name):
        self.rm.viewer.save_screen(dir=dir, name=name)
    def render_ground(self,
                      size=[40.0, 40.0],
                      dsize=[2.0, 2.0], 
                      axis='z',
                      origin=True,
                      use_arrow=True,
                      circle_cut=True):
        if self.rm.tex_id_ground is None:
            self.rm.tex_id_ground = \
              self.rm.gl_render.load_texture(self.rm.file_tex_ground)
        self.rm.gl_render.render_ground_texture(
            self.rm.tex_id_ground,
            size=size,
            dsize=dsize,
            axis=axis,
            origin=origin,
            use_arrow=use_arrow,
            circle_cut=circle_cut)
    def extra_keyboard_callback(self, key):
        pass
    def extra_render_callback(self):
        pass
    def extra_idle_callback(self):
        pass
    def extra_overlay_callback(self):
        pass
    def keyboard_callback(self, key):
        if key in self.rm.toggle:
            self.rm.flag[self.rm.toggle[key]] = not self.rm.flag[self.rm.toggle[key]]
            print('Toggled:', self.rm.toggle[key], self.rm.flag[self.rm.toggle[key]])
        else:
            self.extra_keyboard_callback(key)
    def render_callback(self):
        if self.rm.flag['ground']:
            self.render_ground(
                axis=axis_to_str(self.cam_cur.vup),
                origin=self.rm.flag['origin'])
        self.extra_render_callback()
    def idle_callback(self):
        self.extra_idle_callback()
    def overlay_callback(self):
        if not self.rm.flag['overlay']: return
        self.extra_overlay_callback()
    def update_target_pos(self, pos, ignore_x=False, ignore_y=False, ignore_z=False):
        if np.array_equal(pos, self.cam_cur.origin):
            return
        d = pos - self.cam_cur.origin
        if ignore_x: d[0] = 0.0
        if ignore_y: d[1] = 0.0
        if ignore_z: d[2] = 0.0
        self.cam_cur.translate(d)