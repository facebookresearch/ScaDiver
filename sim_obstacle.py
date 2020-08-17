# Copyright (c) Facebook, Inc. and its affiliates.

from enum import Enum
import numpy as np

from fairmotion.utils import conversions
from fairmotion.processing import operations

from bullet import bullet_utils as bu

import render_module as rm

gl_render = None
bullet_render = None
gl = None

def filter_list_by_index(old_list, pos, positive=True):
    if positive:
        return [old_list[i] for i, e in enumerate(old_list) if i in pos]
    else:
        return [old_list[i] for i, e in enumerate(old_list) if i not in pos]

class Shape(Enum):
    BOX = 1
    SPHERE = 2

class Obstacle(object):
    def __init__(self, name, duration, shape, mass, size, p, Q, v, w):
        self.name = name
        self.duration = duration
        self.mass = mass
        self.size = size
        self.shape = shape
        self.p = p
        self.Q = Q
        self.v = v
        self.w = w
        self.body_id = None
        self.color = [0.8, 0.8, 0.8, 1.0]
        self.lateral_friction = 0.8
        self.spinning_friction = 0.5
        self.restitution = 0.2
        self.linear_damping = 0.0
        self.angular_damping = 0.2
        self.movable = True

class ObstacleManager(object):
    def __init__(self, pb_client, dt, v_up_env, visualization=False):
        self.pb_client = pb_client
        self.obstacles = []
        self.dt = dt
        self.v_up_env = v_up_env
        if visualization:
            global gl_render, bullet_render, gl
            from basecode.render import gl_render
            from basecode.bullet import bullet_render
            import OpenGL.GL as gl
    def clear(self):
        for obs in self.obstacles:
            self.pb_client.removeBody(obs.body_id)
        self.obstacles = []
    def launch(self, obstacle):
        p = self.pb_client
        size = obstacle.size
        if obstacle.shape == Shape.BOX:
            colShapeId = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.5*size[0], 0.5*size[1], 0.5*size[2]])
        elif obstacle.shape == Shape.SPHERE:
            colShapeId = p.createCollisionShape(
                p.GEOM_SPHERE, radius=size[0])
        else:
            raise NotImplementedError
        body_id = p.createMultiBody(baseMass=obstacle.mass,
                                    baseCollisionShapeIndex=colShapeId,
                                    basePosition=obstacle.p,
                                    baseOrientation=obstacle.Q)
        p.resetBaseVelocity(body_id, obstacle.v, obstacle.w)
        p.changeDynamics(body_id, 
                         -1, 
                         lateralFriction=obstacle.lateral_friction, 
                         spinningFriction=obstacle.spinning_friction,
                         restitution=obstacle.restitution,
                         linearDamping=obstacle.linear_damping,
                         angularDamping=obstacle.angular_damping
                         )
        obstacle.body_id = body_id
        self.obstacles.append(obstacle)
    def throw(self, pos_target, num=1, duration=2.0, shape=Shape.BOX, vel=8.0, r_out=2.0, r_in=0.2, mass=2.0, size=0.2*np.ones(3), h_min=0.5):
        assert r_out > r_in
        for _ in range(num):
            d_out = operations.random_unit_vector()
            d_in = operations.random_unit_vector()
            
            p_from = pos_target + r_out * d_out
            
            p_projected_h = operations.projectionOnVector(p_from, self.v_up_env)
            h_cliped = max(np.linalg.norm(p_projected_h), h_min)
            p_from = (p_from-p_projected_h) + h_cliped * self.v_up_env
            p_to = pos_target + r_in * d_in
            v_dir = p_to - p_from
            v_dir = v_dir / np.linalg.norm(v_dir)

            p = p_from
            Q = conversions.A2Q(
                operations.random_unit_vector()*np.random.uniform(-np.pi, np.pi))
            v = vel * v_dir
            w = np.zeros(3)
            obs = Obstacle("", duration, shape, mass, size, p, Q, v, w)
            self.launch(obs)
    def update(self):
        deleted_idx = []
        for i in range(len(self.obstacles)):
            obs = self.obstacles[i]
            obs.duration -= self.dt
            if obs.duration <= 0.0:
                self.pb_client.removeBody(obs.body_id)
                deleted_idx.append(i)
            else:
                p, Q, v, w = bu.get_base_pQvw(self.pb_client, obs.body_id)
                obs.p, obs.Q, obs.v, obs.w = p, Q, v, w
        self.obstacles = filter_list_by_index(self.obstacles, deleted_idx, positive=False)
    def render(self):
        for obs in self.obstacles:
            decay_start = 0.5
            alpha = min(1.0, 1.0/decay_start*obs.duration)
            c = obs.color
            T = conversions.Qp2T(obs.Q, obs.p)
            rm.gl.glPushMatrix()
            rm.gl_render.glTransform(T)
            if obs.shape == Shape.BOX:
                geom_type = self.pb_client.GEOM_BOX
            elif obs.shape == Shape.SPHERE:
                geom_type = self.pb_client.GEOM_SPHERE
            else:
                raise NotImplementedError
            rm.bullet_render.render_geom(geom_type=geom_type, geom_size=obs.size, color=[c[0],c[1],c[2],alpha])
            rm.bullet_render.render_geom_info(geom_type=geom_type, geom_size=obs.size)
            rm.gl.glPopMatrix()
            # bullet_render.render_model(self.pb_client, obs.body_id, color=[0.8, 0.8, 0.8, alpha])
