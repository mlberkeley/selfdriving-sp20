import threading
import json
import os
import numpy as np

import car_iface.controller as controller

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
        if (self.mode == "sysID_test"):
            import car_iface.controller_model as controller_model
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
        if (not (self.mode == "sysID_data" or self.mode == "sysID_test")):
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
            import settings

            cur_dir = os.path.dirname(__file__)
            internal_path = os.path.join(cur_dir, '../../internal_only/sysid/')
            if settings.use_internal_if_available and os.path.exists(internal_path):
                self.data_points_file_path = internal_path
            else:
                self.data_points_file_path = os.path.join(cur_dir, '../hw/sysid/')
            self.data_points_file_path = os.path.join(self.data_points_file_path, f'{car_model}_data_points.json')


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
        if (self.mode == "sysID_test"):
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

            if (self.mode == "sysID_test"):
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

        if (self.mode == "sysID_test"):
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

        if (self.mode == "sysID_data" or self.mode == "sysID_test"):
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

        if (self.mode == "sysID_test"):
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
        if (self.mode == "sysID_test"):
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
