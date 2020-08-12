from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import pybullet
from fairmotion.viz import gl_render
from fairmotion.utils import conversions
from fairmotion.utils import constants


def glTransform(quaternion):

    glMultMatrixd(T.transpose().ravel())

class RenderOptionBody():
    def __init__(self):
        self.render = True
        self.render_edge = True
        self.render_face = True
        self.color_edge = [0.0, 0.0, 0.0, 1.0]
        self.color_face = [0.5, 0.5, 1.0, 0.5]
        self.line_width = 1.0
        self.scale = 1.0
        self.lighting = False


class RenderOptionJoint():
    def __init__(self):
        self.render = False
        self.render_pos = True
        self.render_ori = True
        self.color_pos = [0, 0, 0, 1.0]
        self.color_ori = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.line_width = 1.0
        self.scale = 1.0
        self.lighting = False

class RenderOption():
    def __init__(self):
        self.body = RenderOptionBody()
        self.joint = RenderOptionJoint()

def render_joint(joint, option):
    glPushAttrib(GL_LIGHTING)
    if option.lighting:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)
    j_type = joint.type()
    T0 = joint.child_bodynode.transform()
    T1 = joint.transform_from_child_body_node()
    T = T0.dot(T1)
    R, p = conversions.T2Rp(T)

    scale = option.scale

    if option.render_pos:
        gl_render.render_point(p,
                               color=option.color_pos,
                               scale=scale,
                               radius=0.1)

    if option.render_ori:
        if j_type == "WeldJoint":
            pass
        elif j_type == "RevoluteJoint":
            axis1 = joint.axis_in_world_frame()
            gl_render.render_line(p,
                                  p + scale * axis1,
                                  color=option.color_ori[0])
        elif j_type == "UniversalJoint":
            axis1 = joint.axis1_in_world_frame()
            axis2 = joint.axis2_in_world_frame()
            gl_render.render_line(p,
                                  p + scale * axis1,
                                  color=option.color_ori[0])
            gl_render.render_line(p,
                                  p + scale * axis2,
                                  color=option.color_ori[1])
        elif j_type == "EulerJoint":
            gl_render.render_transform(T,
                                       scale=scale,
                                       color_pos=option.color_pos,
                                       color_ori=option.color_ori)
        elif j_type == "BallJoint":
            gl_render.render_transform(T,
                                       scale=scale,
                                       color_pos=option.color_pos,
                                       color_ori=option.color_ori)
        elif j_type == "TranslationalJoint":
            gl_render.render_transform(T,
                                       scale=scale,
                                       color_pos=option.color_pos,
                                       color_ori=option.color_ori)
        elif j_type == "FreeJoint":
            gl_render.render_transform(T,
                                       scale=scale,
                                       color_pos=option.color_pos,
                                       color_ori=option.color_ori)
        else:
            raise NotImplementedError("Not Implemented")
    glPopAttrib(GL_LIGHTING)


def get_wing_info(world, bid1, bid2, ntype):
    body1 = world.skel.body(bid1)
    body2 = world.skel.body(bid2)
    joint1 = body1.parent_joint
    joint2 = body2.parent_joint

    p1 = joint1.position_in_world_frame()
    c1 = body1.com()
    p1_end = p1 + 2 * (c1 - p1)

    p2 = joint2.position_in_world_frame()
    c2 = body2.com()
    p2_end = p2 + 2 * (c2 - p2)

    if ntype == 0:
        v1 = p1_end
        v2 = p2_end
        v3 = 0.5 * (p1 + p2)
    elif ntype == 1:
        v1 = p2_end
        v2 = p1_end
        v3 = 0.5 * (p1 + p2)
    elif ntype == 2:
        v1 = p2_end
        v2 = p1
        v3 = 0.5 * (p1_end + p2)
    elif ntype == 3:
        v1 = p1
        v2 = p2_end
        v3 = 0.5 * (p1_end + p2)
    else:
        raise NotImplementedError("Not Implemented")

    c = 0.333333 * (v1 + v2 + v3)
    n = np.cross(v1 - v3, v2 - v3)
    n = n / np.linalg.norm(n)

    return v1, v2, v3, c, n


def render_wing_force_one_segment(wing_info, force, option):
    v1, v2, v3, c, n = wing_info
    if option.wing.render_force:
        gl_render.render_line(c,
                              c + 0.01 * force,
                              color=option.wing.color_force)


def render_wing_one_segment(wing_info, option):
    v1, v2, v3, c, n = wing_info
    if option.wing.render_face:
        gl_render.render_tri(v3, v1, v2, color=option.wing.color_face)
    if option.wing.render_normal:
        gl_render.render_line(c, c + n, color=option.wing.color_normal)


def render_wing(world, option):
    glPushAttrib(GL_LIGHTING)
    if option.wing.lighting:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)

    for bid1, bid2, ntype in world.aero_flesh:
        wing_info = get_wing_info(world, bid1, bid2, ntype)
        render_wing_one_segment(wing_info, option)
    for bid1, bid2, ntype, f in world.render_temp:
        wing_info = get_wing_info(world, bid1, bid2, ntype)
        render_wing_force_one_segment(wing_info, f, option)

    glPopAttrib(GL_LIGHTING)

def get_visual_aspects(skel):
    vas = []
    for body in skel.bodynodes:
        for shape in body.shapenodes:
            vas.append(shape.visual_aspect_rgba())
    return vas

def set_visual_aspect(skel, vas):
    cnt = 0
    for body in skel.bodynodes:
        for shape in body.shapenodes:
            c = vas[cnt]
            shape.set_visual_aspect_rgba(c)
            cnt += 1

def get_visual_aspects_body(body):
    vas = []
    for shape in body.shapenodes:
        vas.append(shape.visual_aspect_rgba())
    return vas

def set_visual_aspect_body(body, vas):
    cnt = 0
    for shape in body.shapenodes:
        c = vas[cnt]
        shape.set_visual_aspect_rgba(c)
        cnt += 1

def render_body(body, option, global_coord=True):
    T = body.parent_bodynode.T if body.parent_bodynode else constants.eye_T()

    glPushAttrib(GL_LIGHTING)
    if option.lighting:
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)
    if option.render_face:
        s = option.scale
        glPushMatrix()
        if global_coord:
            gl_render.glTransform(T)
        glScalef(s, s, s)
        body.render_with_color(option.color_face)
        glPopMatrix()
    if option.render_edge:
        glLineWidth(option.line_width)
        s = option.scale
        glPushMatrix()
        if global_coord:
            gl_render.glTransform(T)
        glScalef(s, s, s)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        body.render_with_color(option.color_edge)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()
    glPopAttrib(GL_LIGHTING)

def render_skeleton(skel, option):
    if option is None:
        option = RenderOption()
    if option.body.render:
        # skel.render_with_color(option.body.color_face)
        for body in skel.bodynodes:
            render_body(body, option.body)
    if option.wing.render:
        render_wing(world, option)
    if option.joint.render:
        for joint in skel.joints:
            render_joint(joint, option.joint)


def render_world(world, option=None):
    if option is None:
        option = RenderOption()
    for skel in world.skeletons:
        render_skeleton(skel, option)

# def render_joint_with_body(joint):
#     T0 = joint.child_bodynode.transform()
#     T1 = joint.transform_from_child_body_node()


# def render_body(body):
#     T0 = body.transform()
#     body.shapenodes[0]

# def render_skeleton(skel):
#     for joint in skel.joints:
#         body = joint.child_bodynode
#         T0 = joint.child_bodynode.transform()
#         T1 = joint.transform_from_child_body_node()

def render_geom_bounding_box(geom_type, geom_size, color=[0, 0, 0, 1], T=constants.EYE_T):
    if geom_type==pybullet.GEOM_SPHERE:
        size = [2*geom_size[0], 2*geom_size[0], 2*geom_size[0]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    elif geom_type==pybullet.GEOM_BOX:
        size = [geom_size[0], geom_size[1], geom_size[2]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    elif geom_type==pybullet.GEOM_CAPSULE:
        size = [2*geom_size[1], 2*geom_size[1], 2*geom_size[1]+geom_size[0]]
        gl_render.render_cube(T, size=size, color=color, solid=False)
    else:
        raise NotImplementedError()

def render_geom(geom_type, geom_size, color=[0.5, 0.5, 0.5, 1.0], T=constants.EYE_T):
    if geom_type==pybullet.GEOM_SPHERE:
        gl_render.render_sphere(T, geom_size[0], color=color, slice1=16, slice2=16)
    elif geom_type==pybullet.GEOM_BOX:
        gl_render.render_cube(T, size=[geom_size[0], geom_size[1], geom_size[2]], color=color)
    elif geom_type==pybullet.GEOM_CAPSULE:
        gl_render.render_capsule(T, geom_size[0], geom_size[1], color=color, slice=16)
    elif geom_type==pybullet.GEOM_CYLINDER:
        gl_render.render_cylinder(T, geom_size[0], geom_size[1], color=color, slice=16)
    # elif geom_type==pybullet.GEOM_PLANE:
    #     pass
    # elif geom_type==pybullet.GEOM_MESH:
    #     pass
    else:
        raise NotImplementedError()

def render_geom_info(geom_type, geom_size, scale=1.0, color=[0, 0, 0, 1], T=constants.EYE_T, line_width=1.0):
    glPushMatrix()
    glScalef(scale, scale, scale)
    if geom_type==pybullet.GEOM_SPHERE:
        gl_render.render_sphere_info(T, geom_size[0], line_width=line_width, slice=32,  color=color)
    elif geom_type==pybullet.GEOM_BOX:
        gl_render.render_cube(T, size=[geom_size[0], geom_size[1], geom_size[2]], color=color, solid=False, line_width=line_width)
    elif geom_type==pybullet.GEOM_CAPSULE:
        gl_render.render_capsule_info(T, geom_size[0], geom_size[1], line_width=line_width, slice=32, color=color)
    elif geom_type==pybullet.GEOM_CYLINDER:
        gl_render.render_cylinder_info(T, geom_size[0], geom_size[1], line_width=line_width, slice=32, color=color)
    # elif geom_type==pybullet.GEOM_PLANE:
    #     pass
    # elif geom_type==pybullet.GEOM_MESH:
    #     pass
    else:
        raise NotImplementedError()
    glPopMatrix()

def render_links(pb_client, model):
    for j in range(pb_client.getNumJoints(model)):
        link_state = pb_client.getLinkState(model, j)
        p, Q = np.array(link_state[0]), np.array(link_state[1])
        T = conversions.Qp2T(Q, p)
        gl_render.render_point(p, radius=0.01, color=[0, 1, 0, 1])

def render_joints(pb_client, model):
    for j in range(pb_client.getNumJoints(model)):
        joint_info = pb_client.getJointInfo(model, j)
        joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
        T_joint_local = conversions.Qp2T(
            np.array(joint_local_Q), np.array(joint_local_p))
        if link_idx == -1:
            link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_info = pb_client.getLinkState(model, link_idx)
            link_world_p, link_world_Q = link_info[0], link_info[1]
        T_link_world = conversions.Qp2T(
            np.array(link_world_Q), np.array(link_world_p))
        T_joint_world = np.dot(T_link_world, T_joint_local)
        # Render joint position
        glPushMatrix()
        gl_render.glTransform(T_joint_world)
        gl_render.render_point(np.zeros(3), radius=0.02, color=[0, 0, 0, 1])
        # Render joint axis depending on joint types
        joint_type = joint_info[2]
        if joint_type == pb_client.JOINT_FIXED:
            pass
        elif joint_type == pb_client.JOINT_REVOLUTE:
            axis = joint_info[13]
            gl_render.render_line(np.zeros(3), axis, color=[1, 0, 0, 1], line_width=1.0)
        elif joint_type == pb_client.JOINT_SPHERICAL:
            gl_render.render_transform(constants.eye_T(), scale=0.2)
        else:
            raise NotImplementedError()
        glPopMatrix()

def render_joint_geoms(pb_client, model, radius=0.025, color=[0.5, 0.5, 0.5, 1]):
    for j in range(pb_client.getNumJoints(model)):
        joint_info = pb_client.getJointInfo(model, j)
        joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
        T_joint_local = conversions.Qp2T(
            np.array(joint_local_Q), np.array(joint_local_p))
        if link_idx == -1:
            link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_info = pb_client.getLinkState(model, link_idx)
            link_world_p, link_world_Q = link_info[0], link_info[1]
        T_link_world = conversions.Qp2T(
            np.array(link_world_Q), np.array(link_world_p))
        T_joint_world = np.dot(T_link_world, T_joint_local)
        # Render joint position
        glPushMatrix()
        gl_render.render_sphere(T_joint_world, radius, color=color, slice1=16, slice2=16)
        gl_render.render_sphere_info(T_joint_world, radius, line_width=4.0, slice=32)
        glPopMatrix()

def render_model(pb_client, 
                 model, 
                 draw_link=True, 
                 draw_link_info=True, 
                 draw_joint=False, 
                 draw_joint_geom=True, 
                 ee_indices=None, 
                 color=None,
                 link_info_scale=1.0,
                 link_info_color=[0, 0, 0, 1],
                 link_info_line_width=1.0,
                 lighting=True,
                 ):
    if draw_link or draw_link_info:
        data_visual = pb_client.getVisualShapeData(model)
        for d in data_visual:
            if lighting:
                glEnable(GL_LIGHTING)
            else:
                glDisable(GL_LIGHTING)
            link_id = d[1]
            if link_id == -1:
                p, Q = pb_client.getBasePositionAndOrientation(model)
            else:
                link_state = pb_client.getLinkState(model, link_id)
                p, Q = link_state[4], link_state[5]
            p, Q = np.array(p), np.array(Q)
            R = conversions.Q2R(Q)
            T_joint = conversions.Rp2T(R, p)
            T_visual_from_joint = conversions.Qp2T(np.array(d[6]),np.array(d[5]))
            glPushMatrix()
            gl_render.glTransform(T_joint)
            gl_render.glTransform(T_visual_from_joint)
            # if color is None: color = d[7]
            # alpha = 0.5 if draw_joint else color[3]
            if color is None:
                color = [d[7][0], d[7][1], d[7][2], d[7][3]]
            if draw_link: 
                render_geom(geom_type=d[2], geom_size=d[3], color=color)
            if draw_link_info: 
                render_geom_info(geom_type=d[2], geom_size=d[3], color=link_info_color, scale=link_info_scale, line_width=link_info_line_width)
            # if ee_indices is not None and link_id in ee_indices:
            #     render_geom_bounding_box(geom_type=d[2], geom_size=d[3], color=[0, 0, 0, 1])
            glPopMatrix()
    if draw_joint_geom:
        render_joint_geoms(pb_client, model)
    glDisable(GL_DEPTH_TEST)
    if draw_joint:
        render_joints(pb_client, model)
        render_links(pb_client, model)
    glEnable(GL_DEPTH_TEST)

def render_contacts(pb_client, model, scale_h=0.0005, scale_r=0.01, color=[1.0,0.1,0.0,0.5]):
    data = pb_client.getContactPoints(model)
    for d in data:
        p, n, l = np.array(d[6]), np.array(d[7]), d[9]
        p1 = p
        p2 = p + n * l * scale_h
        gl_render.render_arrow(p1, p2, D=scale_r, color=color)