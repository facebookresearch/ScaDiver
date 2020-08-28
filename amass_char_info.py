# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import numpy as np

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([0.0, 1.0, 0.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([0.0, 0.0, 1.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 0.0, 1.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18

''' 
Definition of the root (base) joint
'''
ROOT = root

''' 
Definition of end effectors
'''
end_effector_indices = [
    lwrist, rwrist, lankle, rankle,
]

''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[root] = "root"
joint_name[lhip] = "lhip"
joint_name[lknee] = "lknee"
joint_name[lankle] = "lankle"
joint_name[rhip] = "rhip"
joint_name[rknee] = "rknee"
joint_name[rankle] = "rankle"
joint_name[lowerback] = "lowerback"
joint_name[upperback] = "upperback"
joint_name[chest] = "chest"
joint_name[lowerneck] = "lowerneck"
joint_name[upperneck] = "upperneck"
joint_name[lclavicle] = "lclavicle"
joint_name[lshoulder] = "lshoulder"
joint_name[lelbow] = "lelbow"
joint_name[lwrist] = "lwrist"
joint_name[rclavicle] = "rclavicle"
joint_name[rshoulder] = "rshoulder"
joint_name[relbow] =  "relbow"
joint_name[rwrist] =  "rwrist"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

joint_idx["root"] = root
joint_idx["lhip"] = lhip
joint_idx["lknee"] = lknee
joint_idx["lankle"] = lankle
joint_idx["rhip"] = rhip
joint_idx["rknee"] = rknee
joint_idx["rankle"] = rankle
joint_idx["lowerback"] = lowerback
joint_idx["upperback"] = upperback
joint_idx["chest"] = chest
joint_idx["lowerneck"] = lowerneck
joint_idx["upperneck"] = upperneck
joint_idx["lclavicle"] = lclavicle
joint_idx["lshoulder"] = lshoulder
joint_idx["lelbow"] = lelbow
joint_idx["lwrist"] = lwrist
joint_idx["rclavicle"] = rclavicle
joint_idx["rshoulder"] = rshoulder
joint_idx["relbow"] = relbow
joint_idx["rwrist"] = rwrist


''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lhip] = "lhip"
bvh_map[lknee] = "lknee"
bvh_map[lankle] = "lankle"
bvh_map[rhip] = "rhip"
bvh_map[rknee] = "rknee"
bvh_map[rankle] = "rankle"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[chest] = "chest"
bvh_map[lowerneck] = "lowerneck"
bvh_map[upperneck] = "upperneck"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lshoulder] = "lshoulder"
bvh_map[lelbow] = "lelbow"
bvh_map[lwrist] = "lwrist"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rshoulder] = "rshoulder"
bvh_map[relbow] =  "relbow"
bvh_map[rwrist] =  "rwrist"

''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
bvh_map_inv = collections.OrderedDict()

bvh_map_inv["root"] = root
bvh_map_inv["lhip"] = lhip
bvh_map_inv["lknee"] = lknee
bvh_map_inv["lankle"] = lankle
bvh_map_inv["ltoe"] = None
bvh_map_inv["rhip"] = rhip
bvh_map_inv["rknee"] = rknee
bvh_map_inv["rankle"] = rankle
bvh_map_inv["rtoe"] = None
bvh_map_inv["lowerback"] = lowerback
bvh_map_inv["upperback"] = upperback
bvh_map_inv["chest"] = chest
bvh_map_inv["lowerneck"] = lowerneck
bvh_map_inv["upperneck"] = upperneck
bvh_map_inv["lclavicle"] = lclavicle
bvh_map_inv["lshoulder"] = lshoulder
bvh_map_inv["lelbow"] = lelbow
bvh_map_inv["lwrist"] = lwrist
bvh_map_inv["rclavicle"] = rclavicle
bvh_map_inv["rshoulder"] = rshoulder
bvh_map_inv["relbow"] = relbow
bvh_map_inv["rwrist"] = rwrist

''' 
Definition of PD gains (tuned for Stable PD Controller)
'''
kp = {
    root : 0,
    lhip : 500,
    lknee : 400,
    lankle : 300,
    rhip : 500,
    rknee : 400,
    rankle : 300,
    lowerback : 500,
    upperback : 500,
    chest : 500,
    lowerneck : 200,
    upperneck : 200,
    lclavicle : 400,
    lshoulder : 400,
    lelbow : 300,
    lwrist : 0,
    rclavicle : 400,
    rshoulder : 400,
    relbow : 300,
    rwrist : 0,
    }

kd = {}
for k, v in kp.items():
    kd[k] = 0.1 * v
kd[root] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002


max_force = {
    root : 0,
    lhip : 300,
    lknee : 200,
    lankle : 100,
    rhip : 300,
    rknee : 200,
    rankle : 100,
    lowerback : 300,
    upperback : 300,
    chest : 300,
    lowerneck : 100,
    upperneck : 100,
    lclavicle : 200,
    lshoulder : 200,
    lelbow : 150,
    lwrist : 0,
    rclavicle : 200,
    rshoulder : 200,
    relbow : 150,
    rwrist : 0 ,
    }

''' 
Maximum forces that character can generate when PD controller is used.
'''
contact_allow_map = {
    root : False,
    lhip : False,
    lknee : False,
    lankle : False,
    rhip : False,
    rknee : False,
    rankle : False,
    lowerback : False,
    upperback : False,
    chest : False,
    lowerneck : False,
    upperneck : False,
    lclavicle : False,
    lshoulder : False,
    lelbow : False,
    lwrist : False,
    rclavicle : False,
    rshoulder : False,
    relbow : False,
    rwrist : False,
    }

joint_weight = {
    root : 1.0,
    lhip : 0.5,
    lknee : 0.3,
    lankle : 0.2,
    rhip : 0.5,
    rknee : 0.3,
    rankle : 0.2,
    lowerback : 0.4,
    upperback : 0.4,
    chest : 0.3,
    lowerneck : 0.3,
    upperneck : 0.3,
    lclavicle : 0.3,
    lshoulder : 0.3,
    lelbow : 0.2,
    lwrist : 0.0,
    rclavicle : 0.3,
    rshoulder : 0.3,
    relbow : 0.2,
    rwrist : 0.0,
    }

''' mu, sigma, lower, upper '''
noise_pose = {
    root :      (0.0, 0.1, -0.5, 0.5),
    lhip :      (0.0, 0.1, -0.5, 0.5),
    lknee :     (0.0, 0.1, -0.5, 0.5),
    lankle :    (0.0, 0.1, -0.5, 0.5),
    rhip :      (0.0, 0.1, -0.5, 0.5),
    rknee :     (0.0, 0.1, -0.5, 0.5),
    rankle :    (0.0, 0.1, -0.5, 0.5),
    lowerback : (0.0, 0.1, -0.5, 0.5),
    upperback : (0.0, 0.1, -0.5, 0.5),
    chest :     (0.0, 0.1, -0.5, 0.5),
    lowerneck : (0.0, 0.1, -0.5, 0.5),
    upperneck : (0.0, 0.1, -0.5, 0.5),
    lclavicle : (0.0, 0.1, -0.5, 0.5),
    lshoulder : (0.0, 0.1, -0.5, 0.5),
    lelbow :    (0.0, 0.1, -0.5, 0.5),
    lwrist :    (0.0, 0.1, -0.5, 0.5),
    rclavicle : (0.0, 0.1, -0.5, 0.5),
    rshoulder : (0.0, 0.1, -0.5, 0.5),
    relbow :    (0.0, 0.1, -0.5, 0.5),
    rwrist :    (0.0, 0.1, -0.5, 0.5),
    }

''' mu, sigma, lower, upper '''
noise_vel = {
    root :      (0.0, 0.1, -0.5, 0.5),
    lhip :      (0.0, 0.1, -0.5, 0.5),
    lknee :     (0.0, 0.1, -0.5, 0.5),
    lankle :    (0.0, 0.1, -0.5, 0.5),
    rhip :      (0.0, 0.1, -0.5, 0.5),
    rknee :     (0.0, 0.1, -0.5, 0.5),
    rankle :    (0.0, 0.1, -0.5, 0.5),
    lowerback : (0.0, 0.1, -0.5, 0.5),
    upperback : (0.0, 0.1, -0.5, 0.5),
    chest :     (0.0, 0.1, -0.5, 0.5),
    lowerneck : (0.0, 0.1, -0.5, 0.5),
    upperneck : (0.0, 0.1, -0.5, 0.5),
    lclavicle : (0.0, 0.1, -0.5, 0.5),
    lshoulder : (0.0, 0.1, -0.5, 0.5),
    lelbow :    (0.0, 0.1, -0.5, 0.5),
    lwrist :    (0.0, 0.1, -0.5, 0.5),
    rclavicle : (0.0, 0.1, -0.5, 0.5),
    rshoulder : (0.0, 0.1, -0.5, 0.5),
    relbow :    (0.0, 0.1, -0.5, 0.5),
    rwrist :    (0.0, 0.1, -0.5, 0.5),
    }

collison_ignore_pairs = [

]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight
