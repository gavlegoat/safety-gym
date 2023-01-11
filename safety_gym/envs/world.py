#!/usr/bin/env python

import os
import xmltodict
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import safety_gym

'''
Tools that allow the Safety Gym Engine to interface to MuJoCo.

The World class owns the underlying mujoco scene and the XML,
and is responsible for regenerating the simulator.

The way to use this is to configure a World() based on your needs
(number of objects, etc) and then call `world.reset()`.

*NOTE:* The simulator should be accessed as `world.sim` and not just
saved separately, because it may change between resets.

Configuration is idiomatically done through Engine configuration,
so any changes to this configuration should also be reflected in
changes to the Engine.
'''

# Default location to look for /xmls folder:
BASE_DIR = os.path.dirname(safety_gym.__file__)


def convert(v):
    ''' Convert a value into a string for mujoco XML '''
    if isinstance(v, (int, float, str)):
        return str(v)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(v))


def rot2quat(theta):
    ''' Get a quaternion rotated only about the Z axis '''
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)],
                    dtype='float64')


class World:
    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in
    # `Engine` configuration
    DEFAULT = {
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_xy': np.zeros(2),  # Robot XY location
        'robot_rot': 0,  # Robot rotation about Z axis

        'floor_size': [3.5, 3.5, .1],  # Used for displaying the floor

        # Objects -- this is processed and added by the Engine class
        'objects': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the
        # scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},

        # Determine whether we create render contexts
        'observe_vision': False,
        # Vision observation not currently supported
    }

    def __init__(self, config={}  # , render_context=None
                 ):
        '''
        config - JSON string or dict of configuration.  See self.parse()
        '''
        self.parse(config)  # Parse configuration
        self.first_reset = True
        self.viewer = None
        # self.render_context = render_context
        self.update_viewer_sim = False
        self.robot = Robot(self.robot_base)

    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    def get_sensor(self, name):
        id = self.model.sensor(name).id
        adr = self.model.sensor_adr[id]
        dim = self.model.sensor_dim[id]
        return self.data.sensordata[adr:adr + dim].copy()

    def build(self):  # noqa: C901
        ''' Build a world, including generating XML and moving objects '''
        # Read in the base XML (contains robot, camera, floor, etc)
        self.robot_base_path = os.path.join(BASE_DIR, self.robot_base)
        with open(self.robot_base_path) as f:
            self.robot_base_xml = f.read()
        self.xml = xmltodict.parse(self.robot_base_xml)  # Nested OrderedDict

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move robot position to starting position
        worldbody['body']['@pos'] = \
            convert(np.r_[self.robot_xy, self.robot.z_height])
        worldbody['body']['@quat'] = convert(rot2quat(self.robot_rot))

        # We need this because xmltodict skips over single-item lists in the
        # tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []

        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            # old default rgb1: ".4 .5 .6"
            # old default rgb2: "0 0 0"
            # light pink: "1 0.44 .81"
            # light blue: "0.004 0.804 .996"
            # light purple: ".676 .547 .996"
            # med blue: "0.527 0.582 0.906"
            # indigo: "0.293 0 0.508"
            asset = xmltodict.parse('''
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 0.582 0.906" rgb2="0.1 0.1 0.35"
                        width="800" height="800" markrgb="1 1 1" mark="random" random="0.001"/>
                    <texture name="texplane" builtin="checker" height="100" width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>
                </asset>
                ''')
            self.xml['mujoco']['asset'] = asset['asset']

        # Add light to the XML dictionary
        light = xmltodict.parse('''<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>''')
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse('''
                <geom name="floor" type="plane" condim="6"/>
                ''')
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        for g in worldbody['geom']:
            if g['@name'] == 'floor':
                g.update({'@size': convert(self.floor_size),
                          '@rgba': '1 1 1 1', '@material': 'MatPlane'})

        # Add cameras to the XML dictionary
        cameras = xmltodict.parse('''<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>''')
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation
        # correct)
        theta = self.robot_rot
        xyaxes = dict(
                    x1=np.cos(theta),
                    x2=-np.sin(theta),
                    x3=0,
                    y1=np.sin(theta),
                    y2=np.cos(theta),
                    y3=1
                    )
        pos = dict(
                xp=0*np.cos(theta) + (-2)*np.sin(theta),
                yp=0*(-np.sin(theta)) + (-2)*np.cos(theta),
                zp=2
                )
        track_camera = xmltodict.parse('''<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}" xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>'''.format(**pos, **xyaxes))
        worldbody['body'][0]['camera'] = [
            worldbody['body'][0]['camera'],
            track_camera['b']['camera']
            ]

        # Add objects to the XML dictionary
        for name, object in self.objects.items():
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            object['quat'] = rot2quat(object['rot'])
            if name == 'box':
                dim = object['size'][0]
                object['dim'] = dim
                object['width'] = dim/2
                object['x'] = dim
                object['y'] = dim
                body = xmltodict.parse('''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                    </body>
                '''.format(**{k: convert(v) for k, v in object.items()}))
            else:
                body = xmltodict.parse('''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                    </body>
                '''.format(**{k: convert(v) for k, v in object.items()}))
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name} {object}'
            assert name.replace('mocap', 'obj') in self.objects, f'missing object for {name}'
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap['rot'])
            body = xmltodict.parse('''
                <body name="{name}" mocap="true">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        pos="{pos}" quat="{quat}" contype="0" conaffinity="0" group="{group}"/>
                </body>
            '''.format(**{k: convert(v) for k, v in mocap.items()}))
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse('''
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            '''.format(**{k: convert(v) for k, v in mocap.items()}))
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            geom['quat'] = rot2quat(geom['rot'])
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            body = xmltodict.parse('''
                <body name="{name}" pos="{pos}" quat="{quat}">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}" group="{group}"
                        contype="{contype}" conaffinity="{conaffinity}"/>
                </body>
            '''.format(**{k: convert(v) for k, v in geom.items()}))
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        self.renderer = MujocoRenderer(self.model, self.data, None)

        # Add render contexts to newly created sim
        # if self.render_context is None and self.observe_vision:
        #     render_context = MjRenderContextOffscreen(self.sim, device_id=-1,
        #                                               quiet=True)
        #     render_context.vopt.geomgroup[:] = 1
        #     self.render_context = render_context

        # if self.render_context is not None:
        #     self.render_context.update_sim(self.sim)

        # Recompute simulation intrinsics from new position
        mujoco.mj_step(self.model, self.data)

    def rebuild(self, config={}, state=True):
        ''' Build a new sim from a model if the model changed '''
        if state:
            old_state = self.data
        # self.config.update(deepcopy(config))
        # self.parse(self.config)
        self.parse(config)
        self.build()
        if state:
            self.data = old_state
        mujoco.mj_step(self.model, self.data)

    def reset(self, build=True):
        ''' Reset the world '''
        if build:
            self.build()
        # set flag so that renderer knows to update sim
        self.update_viewer_sim = True

    # def render(self, mode='human'):
    #     ''' Render the environment to the screen '''
    #     if self.viewer is None:
    #         self.viewer = MjViewer(self.sim)
    #         # Turn all the geom groups on
    #         self.viewer.vopt.geomgroup[:] = 1
    #         # Set camera if specified
    #         if mode == 'human':
    #             self.viewer.cam.fixedcamid = -1
    #             self.viewer.cam.type = const.CAMERA_FREE
    #         else:
    #             self.viewer.cam.fixedcamid = self.model.camera_name2id(mode)
    #             self.viewer.cam.type = const.CAMERA_FIXED
    #     if self.update_viewer_sim:
    #         self.viewer.update_sim(self.sim)
    #         self.update_viewer_sim = False
    #     self.viewer.render()

    def robot_com(self):
        '''
        Get the position of the robot center of mass in the simulator world
        reference frame
        '''
        return self.body_com('robot')

    def robot_pos(self):
        '''
        Get the position of the robot in the simulator world reference frame
        '''
        return self.body_pos('robot')

    def robot_mat(self):
        '''
        Get the rotation matrix of the robot in the simulator world reference
        frame
        '''
        return self.body_mat('robot')

    def robot_vel(self):
        '''
        Get the velocity of the robot in the simulator world reference frame
        '''
        return self.body_vel('robot')

    def body_com(self, name):
        '''
        Get the center of mass of a named body in the simulator world reference
        frame
        '''
        i = self.model.body(name).id
        return self.data.subtree_com[i].copy()

    def body_pos(self, name):
        '''
        Get the position of a named body in the simulator world reference frame
        '''
        i = self.model.body(name).id
        return self.data.xpos[i].copy()

    def body_mat(self, name):
        '''
        Get the rotation matrix of a named body in the simulator world
        reference frame
        '''
        i = self.model.body(name).id
        return self.data.xmat[i].copy().reshape((3, 3))

    def body_vel(self, name):
        '''
        Get the velocity of a named body in the simulator world reference frame
        '''
        bi = self.model.body(name).id
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp=jacp, jacr=None, body=bi)
        jacp = jacp.reshape((3, self.model.nv))
        return np.dot(jacp, self.data.qvel)


class Robot:
    ''' Simple utility class for getting mujoco-specific info about a robot '''
    def __init__(self, path):
        base_path = os.path.join(BASE_DIR, path)
        self.model = mujoco.MjModel.from_xml_path(base_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        # Needed to figure out z-height of free joint of offset body
        i = self.model.body('robot').id
        self.z_height = self.data.xpos[i][2]
        # Get a list of geoms in the robot
        tmp_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.geom_names = [n for n in tmp_names if n != 'floor']
        # Needed to figure out the observation spaces
        self.nq = self.model.nq
        self.nv = self.model.nv
        # Needed to figure out action space
        self.nu = self.model.nu
        # Needed to figure out observation space
        # See engine.py for an explanation for why we treat these separately
        self.hinge_pos_names = []
        self.hinge_vel_names = []
        self.ballquat_names = []
        self.ballangvel_names = []
        self.sensor_dim = {}
        sensor_names = [self.model.sensor(i).name
                        for i in range(self.model.nsensor)]
        for name in sensor_names:
            id = self.model.sensor(name).id
            self.sensor_dim[name] = self.model.sensor_dim[id]
            sensor_type = self.model.sensor_type[id]
            if self.model.sensor_objtype[id] == mujoco.mjtObj.mjOBJ_JOINT:
                joint_id = self.model.sensor_objid[id]
                joint_type = self.model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:
                        self.hinge_pos_names.append(name)
                    elif sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL:
                        self.hinge_vel_names.append(name)
                    else:
                        t = self.model.sensor_type[id]
                        raise ValueError(
                            'Unrecognized sensor type {} for joint'.format(t))
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:
                        self.ballquat_names.append(name)
                    elif sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL:
                        self.ballangvel_names.append(name)
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    # Adding slide joints is trivially easy in code, but this
                    # removes one of the good properties about our
                    # observations. (That we are invariant to relative
                    # whole-world transforms) If slide joints are added we
                    # should ensure this stays true!
                    raise ValueError(
                        'Slide joints in robots not currently supported')
