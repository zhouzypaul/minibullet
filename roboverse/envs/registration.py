import gym
import numpy as np
from roboverse.assets.shapenet_object_lists \
    import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, PICK_PLACE_TRAIN_OBJECTS, \
    PICK_PLACE_TEST_OBJECTS, TRAIN_CONTAINERS, TEST_CONTAINERS

ENVIRONMENT_SPECS = (
    {
        'id': 'Widow250Grasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'beer_bottle',
                   'load_tray': True,
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250GraspEasy-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'shed',
                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'load_tray': False,
                   'xyz_action_scale': 0.2,
                   'object_position_high': (.6, .2, -.30),
                   'object_position_low': (.6, .2, -.30),
                   }
    },
    {
        'id': 'Widow250MultiTaskGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250MultiObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,

                   }
    },
    {
        'id': 'Widow250MultiObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 2,

                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,

                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTrain-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': GRASP_TRAIN_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.30),
                   'object_position_low': (.5, .15, -.30),
                   'xyz_action_scale': 0.2,


                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                       (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   }
    },
    {
        'id': 'Widow250MultiThreeObjectGraspTest-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250MultiObjectEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': GRASP_TEST_OBJECTS,
                   'num_objects': 3,

                   'load_tray': False,
                   'object_position_high': (.7, .25, -.30),
                   'object_position_low': (.5, .15, -.30),
                   'xyz_action_scale': 0.2,


                   # Next three entries are ignored
                   'object_names': ('beer_bottle', 'gatorade', 'shed'),
                   'object_scales': (0.7, 0.6, 0.8),
                   'object_orientations': (
                        (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   }
    },
    {
        'id': 'Widow250SingleObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_high': (.68, .25, -.30),
                   'object_position_low': (.53, .15, -.30),
                   'xyz_action_scale': 0.2,
                   }
    },
    # Pick and place environments
    {
        'id': 'Widow250PickPlaceEasy-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.6, .2, -.3),
                   'object_position_high': (.6, .2, -.3),

                   'container_name': 'bowl_small',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PickPlace-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PickPlaceMultiObjectMultiContainerTrain-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectMultiContainerEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': False,
                   'num_objects': 2,

                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'possible_containers': TRAIN_CONTAINERS,

                   }
    },
    {
        'id': 'Widow250PickPlaceMultiObjectMultiContainerTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectMultiContainerEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': False,
                   'num_objects': 2,

                   'possible_objects': PICK_PLACE_TEST_OBJECTS,
                   'possible_containers': TEST_CONTAINERS,
                   }
    },
    {
        'id': 'Widow250SinglePutInBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250SinglePutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PutInBowlRandomBowlPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTrain-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlRandomBowlPositionTest-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'possible_objects': PICK_PLACE_TEST_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),

                   'container_name': 'bowl_small',


                   }
    },
    {
        'id': 'Widow250PutInTray-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'tray',

                   }
    },
    {
        'id': 'Widow250PutInBox-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'open_box',

                   }
    },
    {
        'id': 'Widow250PlaceOnCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'cube',

                   }
    },
    {
        'id': 'Widow250PutInPanTefal-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalFixedTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   'fixed_container_position': True,
                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   }
    },
    {
        'id': 'Widow250PutInPanTefalTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'pan_tefal',

                   }
    },
    {
        'id': 'Widow250PutInPanTefalRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'pan_tefal',
                   }
    },
    {
        'id': 'Widow250PutInTableTop-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'table_top',

                   }
    },
    {
        'id': 'Widow250PutOnTorus-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'torus',
                   }
    },
    {
        'id': 'Widow250PutOnCubeConcave-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'cube_concave',

                   }
    },
    {
        'id': 'Widow250PutOnPlate-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'plate',

                   }
    },
    {
        'id': 'Widow250PutOnHusky-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'husky',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCube-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,

                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL1Push-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,

                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL1FixedContainerPush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,

                   'container_name': 'marble_cube',
                   'fixed_container_position': True,
                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'marble_cube',

                   }
    },
    {
        'id': 'Widow250PutOnMarbleCubeFixedTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'marble_cube',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutInBasket-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketTestRL1-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',

                   }
    },
    {
        'id': 'Widow250PutInBasketFixedTestRL2-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'sack_vase'),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'shed',

                   'load_tray': False,
                   'container_name': 'basket',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTable-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed', 'two_handled_vase'),
                   'object_scales': (0.7, 0.6),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.69, .27, -.30),

                   'container_name': 'checkerboard_table',

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableFixedTestRL3-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('two_handled_vase',
                                    'thick_wood_chair',),
                   'object_scales': (0.45, 0.4),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'two_handled_vase',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',
                   'fixed_container_position': True,

                   }
    },
    {
        'id': 'Widow250PutOnCheckerboardTableTestRL4-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('curved_handle_cup',
                                    'baseball_cap',),
                   'object_scales': (0.5, 0.5),
                   'object_orientations': ((0, 0.707, 0.707, 0),
                                           (0, -0.707, 0.707, 0)),
                   'target_object': 'curved_handle_cup',

                   'load_tray': False,
                   'container_name': 'checkerboard_table',

                   }
    },
    {
        'id': 'Widow250MultiObjectPutInBowlPush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'bowl_small',
                   }
    },

    {
        'id': 'Widow250MultiObjectPutInBowlPushFixedTrainObj-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('conic_cup', 'fountain_vase'),

                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'bowl_small',
                   }
    },

    {
        'id': 'Widow250MultiObjectPutInBowlPushFixedTestObj-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'object_names': (' b', 'grill_trash_can',),

                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'bowl_small',
                   }
    },

    {
        'id': 'Widow250MultiObjectPutInPlatePush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'plate',
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInCubeConcavePush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'cube_concave',
                   }
    },
    {
        'id': 'Widow250PutInTableTopTestRL1Push-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('square_rod_embellishment',
                                    'grill_trash_can'),
                   'object_scales': (0.6, 0.5),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': 'square_rod_embellishment',
                   'load_tray': False,
                   'container_name': 'table_top_thin',
                   }
    },

    {
        'id': 'Widow250MultiObjectPutInTrayPush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'tray',
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInOpenBoxPush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'open_box',
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInCubePush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'cube',
                   }
    },
    {
        'id': 'Widow250MultiObjectPutInTorusPush-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectEnv',
        'kwargs': {'reward_type': 'push',
                   'control_mode': 'discrete_gripper',
                   'possible_objects': PICK_PLACE_TRAIN_OBJECTS,
                   'num_objects': 2,
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'container_name': 'torus',
                   }
    },
    # Drawer environments
    {
        'id': 'Widow250DrawerOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerEnv',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'start_opened': True,
                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerRandomizedOpen-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerRandomizedEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball',),
                   'object_scales': (0.75,),
                   'target_object': 'ball',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250DrawerRandomizedOpenTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_drawer:Widow250DrawerRandomizedEnv',
        'kwargs': {'reward_type': 'opening',
                   'control_mode': 'discrete_gripper',

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
    # Button environments
    {
        'id': 'Widow250ButtonPress-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250ButtonPressTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'object_position_high': (.75, .25, -.30),
                   'object_position_low': (.6, .1, -.30),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
    {
        'id': 'Widow250RandPosButtonPressTwoObjGrasp-v0',
        'entry_point': 'roboverse.envs.widow250_button:Widow250ButtonEnv',
        'kwargs': {'control_mode': 'discrete_gripper',
                   'button_pos_low': (0.5, 0.25, -.34),
                   'button_pos_high': (0.55, 0.15, -.34),

                   'object_names': ("shed", "sack_vase"),
                   'object_scales': (0.6, 0.6),
                   'object_orientations': ((0, 0, 1, 0), (0, 0.707, 0.707, 0)),
                   'object_position_high': (.75, .25, -.30),
                   'object_position_low': (.65, .1, -.30),
                   'target_object': "shed",
                   'load_tray': False,
                   }
    },
    # ObjectEnv
    {
        'id': 'PickPlaceTrainObject-v0',
        'entry_point': 'roboverse.envs.objects_env:ObjectsEnv',
        'kwargs': {
                   }
    },
    {
        'id': 'PutBottleintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('gatorade', 'bullet_vase'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'gatorade',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'PutBallintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'PutAerointoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('aero_cylinder', 'conic_bin'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'aero_cylinder',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Putbeehive_funnelintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('beehive_funnel', 'toilet_bowl'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'beehive_funnel',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Putpacifier_vaseintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('pacifier_vase', 'colunnade_top'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'pacifier_vase',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Putcrooked_lid_trash_canlintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('crooked_lid_trash_can', 'pepsi_bottle'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'crooked_lid_trash_can',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Puttongue_chairintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('tongue_chair', 'bongo_drum_bowl'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'tongue_chair',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Putvintage_canoeintoBowl-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('vintage_canoe', 't_cup'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'vintage_canoe',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'PutBallintoBowlFixed-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.54, .22, -.30),
                   'object_position_high': (.54, .22, -.30),
                   'container_name': 'bowl_small',
                    # 'fixed_container_position': True,
                   }
    },
    {
        'id': 'PutBallintoBowlCamera-v0',
        'entry_point': 'roboverse.envs.random_camera_view:Widow250MultViewPickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'PutBallintoBowlHalf-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   'config_type': 'default_half',
                   }
    },
    {
        'id': 'PutBallintoBowlThird-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   'config_type': 'default_third',
                   }
    },
    {
        'id': 'PutBallintoBowlDiverse-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),

                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'PutBallintoBowlTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar', 'beer_bottle',),
                   'object_scales': (0.7, 0.7, 0.65),
                   'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   }
    },
    
    {
        'id': 'PutBallintoBowlDiverseLeft-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),
                    'reward_type': 'pick_place_left',
                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'PutBallintoBowlDiverseHalf-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   'config_type': 'diverse_half',
                   }
    },
    {
        'id': 'PutBallintoBowlDiverseThird-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   'config_type': 'diverse_third',
                   }
    },
    {
        'id': 'BinSort-v0',
        'entry_point': 'roboverse.envs.widow250_binsort:Widow250BinSortEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_scales': (0.7, 0.7),
                   'load_tray': False,
                   'object_position_low': (.55, 0.18, -.30),
                   'object_position_high': (.65, 0.37, -.30),
                   'config_type': 'default',
                   'rand_obj':True,
                   }
    },
     {
        'id': 'BinSortTerm-v0',
        'entry_point': 'roboverse.envs.widow250_binsort:Widow250BinSortEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_scales': (0.7, 0.7),
                   'load_tray': False,
                   'object_position_low': (.55, 0.18, -.30),
                   'object_position_high': (.65, 0.37, -.30),
                   'config_type': 'default',
                   'rand_obj':True,
                   'objects_in_container':True,
                   }
    },
    {
        'id': 'BinSortSingleObj-v0',
        'entry_point': 'roboverse.envs.widow250_binsort:Widow250BinSortEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_scales': (0.7, 0.7),
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'config_type': 'default',
                   'rand_obj':True,
                   'num_objects':1,
                   }
    },
    {
        'id': 'BinSortFixedObj-v0',
        'entry_point': 'roboverse.envs.widow250_binsort:Widow250BinSortEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_scales': (0.7, 0.7),
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'config_type': 'default',
                   'rand_obj':False,
                   }
    },
    {
        'id': 'BinSortSingleObjBinObj-v0',
        'entry_point': 'roboverse.envs.widow250_binsort:Widow250BinSortEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_scales': (0.7, 0.7),
                   'load_tray': False,
                   'object_position_low': (.5, .18, -.30),
                   'object_position_high': (.7, .27, -.30),
                   'config_type': 'default',
                   'rand_obj':True,
                   'bin_obj':True, # last object in bin
                   'num_objects':2,
                   }
    },
    {
        'id': 'ReachingSinglePoint-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'reaching_dense',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar'),
                   'object_scales': (0.7, 0.7),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),

                   'container_name': 'bowl_small',
                   'terminate_on_success': False
                   }
    },
    {
        'id': 'Putbullet_vaseintoBowlDiverseTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('bullet_vase', 't_cup', 'pepsi_bottle'),
                   'object_scales': (-1, -1, -1),
                   'target_object': 'bullet_vase',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                    'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),

                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'Putglass_half_gallonintoBowlDiverseTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('glass_half_gallon', 'flat_bottom_sack_vase', 'beehive_funnel'),
                   'object_scales': (-1, -1, -1),
                   'target_object': 'glass_half_gallon',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                    'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),

                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'PutT_cupintoBowlDiverseTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('t_cup', 'ball', 'toilet_bowl'),
                   'object_scales': (-1, -1, -1),
                   'target_object': 't_cup',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                    'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'Putcolunnade_topintoBowlDiverseTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('colunnade_top', 'aero_cylinder', 'hex_deep_bowl'),
                   'object_scales': (-1, -1, -1),
                   'target_object': 'colunnade_top',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                    'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'PutBallintoBowlDiverseTwoDistractors-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('ball', 'jar', 'beer_bottle'),
                   'object_scales': (0.7, 0.7, 0.65),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                    'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
                   'container_name': 'bowl_small',
                   'config_type': 'diverse',
                   }
    },
    {
        'id': 'PutBallintoBowlDiverseBallPosition-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',
                   'object_names': ('ball', ),
                   'object_scales': (0.7, ),
                   'target_object': 'ball',
                   'load_tray': False,
                   'object_position_low': (.47, .15, -.30),
                   'object_position_high': (.69, .3, -.30),
                   'container_name': 'bowl_small',
                   }
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in ENVIRONMENT_SPECS)

    return gym_ids


def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
