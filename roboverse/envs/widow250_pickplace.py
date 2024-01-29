from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
# from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.envs.multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import *

import numpy as np

class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 config_type='default',
                 **kwargs
                 ):
        self.container_name = container_name

        ct = None
        if config_type == 'default':
            ct = CONTAINER_CONFIGS
        elif config_type == 'default_half':
            ct = CONTAINER_CONFIGS_HALVED
        elif config_type == 'default_third':
            ct = CONTAINER_CONFIGS_THIRD
        elif config_type == 'diverse':
            ct = CONTAINER_CONFIGS_DIVERSE
        elif config_type == 'diverse_half':
            ct = CONTAINER_CONFIGS_DIVERSE_HALVED
        elif config_type == 'diverse_third':
            ct = CONTAINER_CONFIGS_DIVERSE_THIRD
        else:
            assert False, 'Invalid config type'

        container_config = ct[self.container_name]
        print('Container config:', container_config)

        self.fixed_container_position = fixed_container_position
        if self.fixed_container_position:
            self.container_position_low = container_config['container_position_default']
            self.container_position_high = container_config['container_position_default']
        else:
            self.container_position_low = container_config['container_position_low']
            self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']

        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']
        # self.push_success_radius_threshold = container_config['push_success_radius_threshold']

        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self, load_object_positions=None, load_container_position=None):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        """
        TODO(avi) This needs to be cleaned up, generate function should only 
                  take in (x,y) positions instead. 
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if not self.in_vr_replay:
            # if self.num_objects >= 2:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                    num_small_obj=self.num_objects
                )
            # elif self.num_objects == 1:
            #     self.container_position, self.original_object_positions = \
            #         object_utils.generate_object_positions_single(
            #             self.object_position_low, self.object_position_high,
            #             self.container_position_low, self.container_position_high,
            #             min_distance_large_obj=self.min_distance_from_object,
            #         )
            # else:
            #     raise NotImplementedError

        # TODO(avi) Need to clean up
        if load_container_position is None:
            container_position = self.container_position
        else:
            container_position = load_container_position
        container_position[-1] = self.container_position_z

        self.x_diff = self.container_position_high[0] - container_position[0]
        self.iscontainerleft = abs(self.x_diff) < 0.1
        self.container_id = object_utils.load_object(self.container_name,
                                                     container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        object_positions = load_object_positions or self.original_object_positions
        
        for object_name, object_position in zip(self.object_names, object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self, object_positions=None, container_position=None):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes(
            load_object_positions=object_positions, 
            load_container_position=container_position
        )
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True  # TODO(avi): Clean this up

        return self.get_observation()

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'pick_place_left':
            reward = float(info['place_success_target_left'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        elif self.reward_type == 'push':
            reward = float(info['push_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        info['place_success_target_left'] = info['place_success_target'] and self.iscontainerleft
        
        info['object_positions'] = self.original_object_positions
        info['container_position'] = self.container_position

        if self.reward_type == 'push':
            info['push_success'] = False
            for object_name in self.object_names:
                push_success = object_utils.check_near_container(
                    object_name, self.objects, self.container_position,
                    self.push_success_radius_threshold
                )
                if push_success:
                    info['push_success'] = push_success

            info['push_success_target'] = object_utils.check_near_container(
                self.target_object, self.objects, self.container_position,
                self.push_success_radius_threshold)
        return info


class Widow250PickPlaceMultiObjectEnv(MultiObjectEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceMultiObjectMultiContainerEnv(
    MultiObjectMultiContainerEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":

    # Fixed container position
    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed', 'two_handled_vase'),
        object_scales=(0.7, 0.6),
        target_object='shed',
        load_tray=False,
        object_position_low=(.49, .18, -.20),
        object_position_high=(.59, .27, -.20),

        container_name='cube',
        container_position_low=(.72, 0.23, -.20),
        container_position_high=(.72, 0.23, -.20),
        container_position_z=-0.34,
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.05,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    # env = Widow250PickPlaceEnv(
    #     reward_type='pick_place',
    #     control_mode='discrete_gripper',
    #     object_names=('shed',),
    #     object_scales=(0.7,),
    #     target_object='shed',
    #     load_tray=False,
    #     object_position_low=(.5, .18, -.25),
    #     object_position_high=(.7, .27, -.25),
    #
    #     container_name='bowl_small',
    #     container_position_low=(.5, 0.26, -.25),
    #     container_position_high=(.7, 0.26, -.25),
    #     container_orientation=(0, 0, 0.707107, 0.707107),
    #     container_scale=0.07,
    #
    #     camera_distance=0.29,
    #     camera_target_pos=(0.6, 0.2, -0.28),
    #     gui=True
    # )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)
