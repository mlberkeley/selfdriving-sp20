#!/usr/bin/env python3

"""
Simulator we will be using to interact with the duckietown environment.
Note: Control-C to exit simulator.
"""
import sys

import time
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import threading
import os
import json
import matplotlib.pyplot as plt

sys.path.append('duckietown-sim')
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper



sys.path.append('Car Interface Weeks 2-3')
import controller

'''
The Car class is used to encapsulate the car entity as
it drives around Duckietown.  It keeps track of the internal
dynamics and provides the current state of the car based
on previous user inputs.
'''
class Car():

    '''
    mode corresponds to what mode the simulator is being run in (ex. sysID_data)
    Depending on the mode the user inputs and terminal outputs will vary.

    car_model either "simple" or "complex".  Represents the internal dynamics of
    the car as either linear based or non linear.
    '''
    def __init__(self, mode, car_model = "simple"):

        self.mode = mode
        self.car_model = car_model

        '''
        Initialize the interface that will be used to manage motion dynamics.
        '''
        self.interface = controller.Car_Interface(car_model)

        '''
        If we are testing system ID then we will initialize an instance of the
        controller model, which we will interact with side by side along with
        the actual interface.
        '''
        if (self.mode == "sysID_testing"):
            self.interface_model = controller_model.Car_Interface(car_model)

        '''
        pedal_type intitialized to None
        peda_type can later be "accelerate", "brake", or "release"

        amount is relevant for both "accelerate" and "brake" and
        represents the magnitude of pedal depression.

        TIME_UNIT is a constant and represents how often we update
        the car's state.
        '''
        self.pedal_type = None
        self.amount = 0.0
        self.TIME_UNIT = self.interface.dt

        '''
        Build up is used to allow for increasing pedal depression
        the longer you hold a control.  That is when you press as
        you hold "accelerate" or "brake" the pedal depression will
        increase from 0 to 1 over time.

        When we are doing SystemID we will not follow the build up
        principle and apply random depression quantities, to make sure
        we are covering all possible scenarios.
        '''
        if (not (self.mode == "sysID_data" or self.mode == "sysID_testing")):
            self.build_up = 0
            self.bu_to_amt = lambda : 1 - 1 / ((self.build_up / 20) + 1)

        '''
        If we are collecting data for System ID we intialize a file path for
        where we will save the data points (including reference to the internal
        car model).
        If there is a preexisting data points file we will load it
        and add to it, as opposed to starting from scratch.
        We will also keep track how long ago we last saved the collected data.
        '''
        if (self.mode == "sysID_data"):
            self.data_points_file_path = f"Car Interface Weeks 2-3/SystemID/{car_model}_data_points.json"

            if (os.path.exists(self.data_points_file_path)):
                self.data_points = json.load(open(self.data_points_file_path, "r"))
            else:
                self.data_points = []

            self.saved_data = 0

        '''
        If we are testing System ID we will keep track of a rolling average of
        the discrepancy between the actual controller and the controller model.

        The average will be taken over the last ERROR_WINDOW times.  The maximum
        average error will be tracked as well.
        '''
        if (self.mode == "sysID_testing"):
            self.ERROR_WINDOW = 100
            self.errors = []
            self.max_mean_error = None

    '''
    Start's the car.  Every TIME_UNIT we update the state of the car in accordance
    to it's motion dynamics.  The state of the car most importantly contains the
    velocity and steering angle.
    '''
    def start(self):

        threading.Timer(self.TIME_UNIT, self.update_state).start()

    '''
    Updates the state of the car by applying the current control
    being provided by the user to the controlller.
    '''
    def update_state(self):

        #Measure the previous velocity to be used in motion dynamics.
        previous_velocity = self.interface.measure("velocity")

        '''
        if gear is None there is no motion.  Otherwise we apply the
        currently user provided control to the controller.  If we
        are testing system ID we apply the same control to our
        controller model.
        '''
        if (self.interface.get_gear() is not None):
            self.interface.apply_control(self.pedal_type, self.amount)

            if (self.mode == "sysID_testing"):
                self.interface_model.apply_control(self.pedal_type, self.amount)

        '''
        This section is for System ID data collection.  For simplicity we only collect when
        moving forward.
        The data point to be saved is (pedal_name, amount, previous_vel, accel).

        pedal_name: "None", "accelerate", or "brake" accordingly (None is neither pedal)
        amount: pedal depression 0 to 1 (only nonzero if pedal_name is "accelerate" or "brake")
        previous_vel: 0 to 1, measured previous velocity (has gaussian noise)
        accel: 0 to 1, measured acceleration (has gaussian noise)

        Displays the current state of the car as well.
        Increments the saved_data counter which represents how long ago we saved data.
        '''
        if (self.mode == "sysID_data" and self.interface.get_gear() == self.interface.FORWARD):
            pedal_name = None
            if (self.pedal_type == self.interface.ACCELERATOR):
                pedal_name = "accelerator"
            elif (self.pedal_type == self.interface.BRAKE):
                pedal_name = "brake"

            accel = self.interface.measure("accel")
            data_point = (pedal_name, self.amount, previous_velocity, accel)
            print(f"{pedal_name} {self.amount * 100:.2f}%, VEL: {previous_velocity * 100:.2f}%, ACCEL: {accel * 100:.3f}%/s           ", end = "\r")
            self.data_points.append(data_point)

            self.saved_data += 1

        #Will call same method again in TIME_UNIT seconds.
        threading.Timer(self.TIME_UNIT, self.update_state).start()

    '''
    Sets gear accordingly ("forward" or "reverse")
    If testing System ID makes the corresponding number to model.
    '''
    def gear(self, g):
        gear = None
        if (g == "forward"):
            gear = self.interface.FORWARD
        elif (g == "reverse"):
            gear = self.interface.REVERSE

        self.interface.set_gear(gear)

        if (self.mode == "sysID_testing"):
            self.interface_model.set_gear(gear)

    '''
    Sets pedal accordingly ("accelerate", "brake", or "release")
    If same pedal as before increment pedal depression amount build up
    If different pedal then reset build up to 0.

    If not doing System ID then set the amount in accordance to the build up.

    If doing System ID and the pedal is accelerator or brake then set it to
    a random number between 0 and 1.  This is to create a more diverse set
    of scenearios.  (Can be thought of as randomly depressing and releasing
    accelerator or brake)

    Note: "release" or no pedal will always have 0 depression amount
    '''
    def pedal(self, p):

        if (p == "accelerate"):
            if (self.pedal_type == self.interface.ACCELERATOR):
                self.build_up += 1
            else:
                self.pedal_type = self.interface.ACCELERATOR
                self.build_up = 0

        elif (p == "brake"):
            if (self.pedal_type == self.interface.BRAKE):
                self.build_up += 1
            else:
                self.pedal_type = self.interface.BRAKE
                self.build_up = 0

        elif (p == "release"):
            self.pedal_type = None
            self.build_up = 0

        if (self.mode == "sysID_data" or self.mode == "sysID_testing"):
            if (self.pedal_type is None):
                self.amount = 0
            else:
                self.amount = np.random.random()
        else:
            self.amount = self.bu_to_amt()

    '''
    custom_pedal lets you directly set the depression amount in addition
    to the desired pedal.  This circumvents the build up default procedure.
    '''
    def custom_pedal(self, p, amt):
        if (p == "accelerate"):
            self.pedal_type = self.interface.ACCELERATOR
        elif (p == "brake"):
            self.pedal_type = self.interface.BRAKE
        elif (p == "release"):
            self.pedal_type = None
        self.amount = amt

    '''
    Sets steering angle based on user input.  By default steers to full
    capacity left or right or not at all based on user input.

    Passes input to the controller and controller model as well if doing
    System ID testing.
    '''
    def turn(self, t):
        if (t == "left"):
            steer_angle = 1.0
        elif (t == "right"):
            steer_angle = -1.0
        elif (t == "release"):
            steer_angle = 0.0

        self.interface.steer_to(steer_angle)

        if (self.mode == "sysID_testing"):
            self.interface_model.steer_to(steer_angle)

    '''
    Resets the car state by re initializing (typically after a crash).
    If doing System ID data collections saves before resetting to prevent
    loss of data.
    '''
    def reset(self):
        if (self.mode == "sysID_data"):
            self.save_data()
        self.__init__(self.mode, self.car_model)

    '''
    Returns the duckietown control (velocity, steering angle) as prescribed
    by the controller module.
    If testing System ID will update model discrepancies and present the rolling
    average to the user.  If the rolling average ever exceeds the 0.75% threshold will
    raise and Exception.
    Note: Model discrepancy is measured based on velocity and not acceleration.  This
    is to capture the potential build up of error.  In addition, noise in measurement
    will contribute to the error, but it will only be about 0.4% (below the 0.75% threshold).
    '''
    def duckietown_control(self):

        if (self.mode == "sysID_testing"):
            measured = self.interface.measure("velocity")
            predicted = self.interface_model.velocity
            error = abs(measured - predicted)
            if (len(self.errors) == self.ERROR_WINDOW):
                self.errors = self.errors[1:] + [error]
                mean_error = sum(self.errors)/self.ERROR_WINDOW
                if (self.max_mean_error is None or mean_error > self.max_mean_error):
                    self.max_mean_error = mean_error

                print(f"WINDOW MEAN ERROR {mean_error * 100:.2f}%; MAX MEAN ERROR {self.max_mean_error * 100:.2f}%", end = "\r")
                if (self.car_model == "simple"):
                    if (mean_error > 0.0075):
                        raise Exception(f"Interface Model Error Exceeds 0.75% Threshold")
                elif (self.car_model == "complex"):
                    if (mean_error > 0.02):
                        raise Exception(f"Interface Model Error Exceeds 2% Threshold")
            else:
                print("BUILDING PRELIMINARY WINDOW..." + " " * 10, end = "\r")
                self.errors.append(error)

        return self.interface.duckietown_control()

    '''
    Only used if collecting System ID data.  If you have not
    saved in the last 5 time steps than save the currently tracked
    data points.  Further, reset the saved data counter.
    '''
    def save_data(self):
        if (self.mode == "sysID_data" and self.saved_data >= 5):
            json.dump(self.data_points, open(self.data_points_file_path, "w"))
            print("\nSaved System ID data")
            self.saved_data = 0

'''
Command line arguments only two you need to use are --mode and --car_model.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--mode', default = 'manual_control')
parser.add_argument('--car_model', default = 'simple')
args = parser.parse_args()

'''
Mode should be set depending on what work you are currently doing with the simulator.
The valid options are shown below.

manual_control: Default.  Nothing assignment specific, just driving around simulator.
sysID_data: System ID Data Collection.  Click 's' to save data points.
sysID_testing: System ID Testing.  Keeps track of actual and model discrepancy.
braking_distance:  Braking Distance.  Click 'm' to mark a braking target location.
                   When available, Hold 'b' to stop at the target location ahead.
'''
VALID_MODES = {'manual_control', 'sysID_data', 'sysID_testing', 'braking_distance', 'conv'}
MODE = args.mode
if (MODE not in VALID_MODES):
    raise Exception(f"Provided mode, {MODE}, not in {VALID_MODES}")

'''
Either "simple" or "complex" representing internal model dynamics.
'''
CAR_MODEL = args.car_model
if (CAR_MODEL not in {"simple", "complex"}):
    raise Exception(f"Provided car model {CAR_MODEL}, must be 'simple' or 'complex'")


if (MODE == "sysID_testing"):
    import controller_model


'''
Sets up additional components required for braking distance module.

The Keypoint_Tracker maintains a list of braking target locations, and also
serves to provide the nearest target location in the direction of heading if any.

The brake_log keeps track of the braking inputs used in the current precision
braking attempt.

There are three possible Braking Distance Estimators.

1) fcn, "simple":  FCN for "simple" motion dynamics
2) analytical, "simple": Solves analytical equations exactly for "simple" dynamics.
3) fcn, "complex": FCN for "complex" motion dynamics

All of these given the current velocity and target stopping distance will output
a brake pedal depression amount between 0 and 1.

'''
if (MODE == "braking_distance"):
    sys.path.append('Car Interface Weeks 2-3/BrakingDistance')
    import keypoints
    import bd_api

    kp_tracker = keypoints.Keypoint_Tracker()
    brake_log = []


    if (CAR_MODEL == "complex"):
        estimator_type = "fcn"
    else:
        estimator_type = input("fcn (0) or analytical (1) Braking Distance Estimator? ")
        if (estimator_type not in {'0', '1'}):
            raise Exception("Please input 0 or 1")
        estimator_type = 'fcn' if estimator_type == '0' else 'analytical'

    bd_estimator = bd_api.Braking_Distance_Estimator(CAR_MODEL, estimator_type)

if (MODE == "conv"):
    import preset_convs
    
    topdown_mode = False
    enable_grayscale = False

    cur_conv_n = 0
    convs = [None] + preset_convs.convs

    def conv_filter(img):
        global cur_conv_n
        f = convs[cur_conv_n]
        if f is None:
            return img

        img = img / 255.

        res = [None] * img.shape[2]
        for i in range(img.shape[2]):
            res[i] = preset_convs.conv2d(img[:, :, i], f)
        return np.clip(np.stack(res, axis=2), 0., 1.) * 255.

    def render_filter(img):
        if enable_grayscale:
            img = preset_convs.to_grayscale(img)
        img = conv_filter(img)
        if enable_grayscale:
            img = preset_convs.gs_rgb_repr(img)
        return img.astype('uint8')

    def render():
        env.render(filter=render_filter, top_down=topdown_mode)

env = DuckietownEnv(
    seed = args.seed,
    map_name = args.map_name,
    draw_curve = args.draw_curve,
    draw_bbox = args.draw_bbox,
    domain_rand = args.domain_rand,
    frame_skip = args.frame_skip,
    distortion = args.distortion,
)
env.render()

car = Car(MODE, CAR_MODEL)
car.start()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

if (MODE == "conv"):
    def toggle_grayscale(obs, reward, done, info):
        global enable_grayscale
        enable_grayscale = not enable_grayscale

    def next_conv(obs, reward, done, info):
        global cur_conv_n
        cur_conv_n = (cur_conv_n + 1) % len(convs)

    def save_screenshot(obs, reward, done, info):
        obs = render_filter(obs)

        np.save('screen.npy', obs)

        from PIL import Image
        im = Image.fromarray(obs, mode='RGB')

        im.save('screen.png')

    def save_car_data(obs, reward, done, info):
        car.save_data()

    def toggle_semantic(obs, reward, done, info):
        env.semantic_mode = not env.semantic_mode

    def toggle_mapmode(obs, reward, done, info):
        env.mapmode = not env.mapmode

    def toggle_topdown(obs, reward, done, info):
        global topdown_mode
        topdown_mode = not topdown_mode

    one_shot_keys_down = {}
    action_keys = [
        (key.C, next_conv),
        (key.G, toggle_grayscale),
        (key.ENTER, save_screenshot),
        (key.S, save_car_data),
        (key.M, toggle_semantic),
        (key.J, toggle_mapmode),
        (key.T, toggle_topdown)
    ]


'''
This is the core update function used to handle
keyboard input and update the visual simulation.
'''
def update(dt):

    '''
    For the braking distance module this bit of code gets
    the car's current position and angle from the environment.
    Then it works with the Keypint_Tracker to find if there is
    a nearby target location in the direction of movement and
    if so presents the option to precision brake.
    '''
    if (MODE == 'braking_distance'):
        angle = int(env.cur_angle * 180 / np.pi) % 360
        x, _, y = env.cur_pos

        dist_to = kp_tracker.dist_to_nearest(x, y, angle)
        if (dist_to is not None and dist_to >= 0.01):
            print(f"HOLD 'b' TO BRAKE IN {dist_to:.2f}", end = '\r')
        else:
            print(" " * 30, end = "\r")


    '''
    Set gear to forward or reverse by clicking 'f' or 'r'.
    '''
    if key_handler[key.F]:
        car.gear("forward")
    elif key_handler[key.R]:
        car.gear("reverse")

    '''
    Accelerate or Brake by clicking the up or down arrows.

    If in braking distance mode and a target location is feasible,
    then holding 'b' will attempt a precision brake.
    It will use the Braking Distance Estimator to figure out how
    much to depress the brake at each time step.  Note if you are just at
    or past the target location full brakes will be used.
    Further, it will keep track of the brake depressions being used
    in the brake log.

    Not using any of these controls will translate to releasing the
    pedals.  If in braking distance mode, and just completed a precision brake
    the brake amounts used during the precision brake will be displayed in a graph
    as a function of the distance to goal.
    '''
    if key_handler[key.UP]:
        car.pedal("accelerate")
    elif key_handler[key.DOWN]:
        car.pedal("brake")
    elif MODE == 'braking_distance' and key_handler[key.B] and dist_to is not None:
        if (dist_to > 0):
            brake_amt = bd_estimator.estimate(car.interface.measure("velocity"), dist_to)
        else:
            brake_amt = 1.0
        brake_log.append((dist_to, brake_amt))
        car.custom_pedal("brake", brake_amt)
    else:
        car.pedal("release")

        if (MODE == 'braking_distance' and len(brake_log) > 0):
            dists = [-log[0] for log in brake_log]
            brake_amts = [log[1] for log in brake_log]

            plt.title("Brake Amount vs Distance From")
            plt.xlabel("Distance From")
            plt.ylabel("Brake Amount")
            plt.plot(dists, brake_amts)
            plt.show()

            while(len(brake_log) > 0):
                brake_log.pop()

    '''
    One can turn left or right using the left and right arrows.
    The car moves straight in the absence of either input.
    '''
    if key_handler[key.LEFT]:
        car.turn("left")
    elif key_handler[key.RIGHT]:
        car.turn("right")
    else:
        car.turn("release")

    '''
    Clicking 's' during System ID Data Collection will update
    the data points file.
    Note: quitting the simulator without saving your data will
    lose the data.
    '''
    if MODE == 'sysID_data' and key_handler[key.S]:
        car.save_data()

    '''
    Clicking 'm' during Braking Distance will add a target location.
    One can delete saved target locations by running keypoints.py
    in the Braking Distance folder.
    '''
    if MODE == 'braking_distance' and key_handler[key.M]:
        print(f"Marking (x: {x:.2f}, y: {y:.2f}) angle: {angle} as a keypoint")
        kp_tracker.add_keypoint(x, y, angle)

    #Gets duckietown control (velocity, steering angle) from car instance.
    action = np.array(car.duckietown_control())

    # Need for speed?  Hold the left shift to move 50% faster
    if key_handler[key.LSHIFT] and key_handler[key.UP]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    if (MODE == "conv"):

        for (k, f) in action_keys:
            global one_shot_keys_down
            if key_handler[k]:
                if k not in one_shot_keys_down:
                    one_shot_keys_down[k] = False
                if not one_shot_keys_down[k]:
                    f(obs, reward, done, info)
                    one_shot_keys_down[k] = True
            else:
                one_shot_keys_down[k] = False

    #Reset car instance if crashed
    if done:
        print('\nCrashed!')
        env.reset()
        car.reset()
        env.render()

update_delay = 1.0 / env.unwrapped.frame_rate
next_update = time.time()
def tick(dt):
    t = time.time()

    global next_update
    if t >= next_update:
        update(dt)
        next_update = t + update_delay

    env.render()

pyglet.clock.schedule_interval(tick, 1.0 / 60)

# Enter main event loop
pyglet.app.run()

env.close()
