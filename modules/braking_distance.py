import modules.globals as globals


import logging
log = logging.getLogger('braking_distance')


import numpy as np
from pyglet.window import key
import matplotlib.pyplot as plt

from braking_distance.keypoints import Keypoint_Tracker
from braking_distance.bd_api import Braking_Distance_Estimator

kp_tracker = Keypoint_Tracker()
brake_log = []
dist_to = None
bd_estimator = None

def control_hook():
    global dist_to, brake_log


    key_handler = globals.key_handler
    car = globals.car
    env = globals.env


    angle = int(env.cur_angle * 180 / np.pi) % 360
    x, _, y = env.cur_pos

    dist_to = kp_tracker.dist_to_nearest(x, y, angle)
    in_forward = car.interface.get_gear() == car.interface.FORWARD

    if (dist_to is not None and dist_to >= 0.01 and in_forward):
        print(f"HOLD 'b' TO BRAKE IN {dist_to:.2f}", end = '\r')
    else:
        print(" " * 30, end = "\r")

    if key_handler[key.B] and in_forward:
        if dist_to is None:
            return

        if dist_to > 0:
            brake_amt = bd_estimator.estimate(car.interface.measure("velocity"), dist_to)
        else:
            brake_amt = 1.0

        if (abs(dist_to) < 0.01):
            print(f"AT BRAKE POINT!" + " " * 20, end = "\r")

        brake_log.append((dist_to, brake_amt))
        car.custom_pedal("brake", brake_amt)
    elif brake_log:
        dists = [-log[0] for log in brake_log]
        brake_amts = [log[1] for log in brake_log]

        plt.title("Brake Amount vs Distance From")
        plt.xlabel("Distance From")
        plt.ylabel("Brake Amount")
        plt.plot(dists, brake_amts)
        plt.show()

        brake_log = []

def add_checkpoint(obs, reward, done, info):
    angle = int(globals.env.cur_angle * 180 / np.pi) % 360
    x, _, y = globals.env.cur_pos

    log.info(f"Marking (x: {x:.2f}, y: {y:.2f}) angle: {angle} as a keypoint")
    kp_tracker.add_keypoint(x, y, angle)

# def setup_args(parser):
#     globals.setup_fn('manual', parser)

#     parser.add_argument('--car_model', default='simple')

def register(args):
    car_models = ['simple', 'complex']
    if args.car_model not in car_models:
        models_str = ', '.join([f'"{m}"' for m in car_models])

        log.error(f'Car model "{args.car_model}" not found!')
        log.error(f'Available models are {models_str}')

        import sys
        sys.exit(1)

    globals.enable_fns['manual'](args)

    global bd_estimator

    if args.car_model == "complex":
        estimator_type = "fcn"
    else:
        estimator_type = input("fcn (0) or analytical (1) Braking Distance Estimator? ")
        while estimator_type not in {'0', '1'}:
            print("Please input 0 or 1")
            estimator_type = input("fcn (0) or analytical (1) Braking Distance Estimator? ")

        estimator_type = 'fcn' if estimator_type == '0' else 'analytical'

    bd_estimator = Braking_Distance_Estimator(args.car_model, estimator_type)


    globals.control_hooks += [control_hook]
    globals.keypress_handlers += [
        (key.M, add_checkpoint)
    ]

    log.info('Braking distance module registered')

globals.available_modes += ['braking_distance']
# globals.arg_setup_fns['braking_distance'] = setup_args
globals.enable_fns['braking_distance'] = register
