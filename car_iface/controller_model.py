import os
from utils import nn

class Car_Interface():
    '''
    Intialize car interface object.  Model is default to simple.
    If simple acceleration will roughly be a linear combination
    of the corresponding factors.  model = "complex" introduces
    some nonlinearities
    '''
    def __init__(self, model = "simple"):
        if (model != "simple" and model != "complex"):
            raise Exception(f"Illegal argument model can only be 'simple' or 'complex' not {model}")

        self.model = model

        #Variables to keep track of the car's current state
        self.position = 0
        self.velocity = 0
        self.accel = 0

        self.steering_angle = 0

        self.gear = None

        #Constants corresponding to the car pedals
        self.ACCELERATOR = 0
        self.BRAKE = 1

        #Constants corrsponding to car gears
        self.FORWARD = 0
        self.REVERSE = 1

        '''
        PART OF WEEK 2 HW

        FILL IN ESTIMATED COEFFICIENTS BELOW (Delete exception too)
        All except for the brake_weight should be positive.
        '''
        #Coefficients corresponding to the motion dynamics
        self.rolling_bias = None
        self.friction_constant = None

        self.accelerator_weight = None
        self.brake_weight = None
        raise Exception("You forgot to input SystemID learned weights in the Controller Model")

        '''
        If approximating the complex internal model we use a FCN
        to model the acceleration as a function of the pedal depressions
        and velocity.  Since the weights for the this model will be in the
        SystemID directory we include it in the file path.
        The model has 3 inputs (accelerator depression, brake depression, velocity)
        '''
        if (self.model == "complex"):
            self.complex_accel_fcn = nn.fcn(model_name = self.complex_weights_fp(), num_inputs = 3)

        #Variables to keep track of time (seconds)
        self.T = 0

        #Corresponds to simulations internal discrete time step (15fps)
        self.dt = 1 / 15

    #Depress the specified pedal by the specified amount for one dt time step
    def apply_control(self, pedal, amount):
        if (self.gear is None):
            raise Exception("Please set gear before applying control")

        if (pedal not in [None, self.ACCELERATOR, self.BRAKE]):
            raise Exception(f"Invalid pedal provided, {pedal}")

        if (amount < 0 or amount > 1):
            raise Exception(f"Amount must be between 0 and 1, not {amount}")

        '''
        if the gear is None there is no motion (Parking equivalent)
        '''
        if (self.gear is None):
            return

        '''
        For the "simple" internal model we assume linear dynamics.
        Specifically we model the acceleration as a linear combination
        of the accelerator depression, the brake depression, the velocity,
        and a constant

        For the "complex" internal model we make no assumptions of the
        interanl motion dynamics.  We approximate the dynamics with a learned
        neural network that models accelration as a function of the same inputs
        as before.
        '''
        if (self.model == "simple"):

            '''
            PART OF WEEK 2 HW
                                 Part A                     Part B
            accel = [c_1 * accel_amt + c_2* brake_amt] + [c_3 * v + c_4]

            c_1: accelerator_weight
            c_2: brake_weight
            c_3: -friction_constant
            c_4: rolling_bias
            (Remember the self prefix for these parameters)

            Implementation instructions:
            PART A:
             a.accel_amt = 0, brake_amt = 0, if pedal is None
             b.accel_amt = amount, brake_amt = 0, if pedal is self.ACCELERATOR
             c.accel_amt = 0, brake_amt = amount, if pedal is self.BRAKE
            Part B:
             a.Use the absolute value of the current velocity for v.

            self.accel should be set to the sum of these components.
            '''

            #CODE HERE (Delete exception too)
            raise Exception("You forgot to fill Simple Acceleration Calcs in the Controller Model")

        elif (self.model == "complex"):
            '''
            PART OF WEEK 3 HW

            Here we will use our fully connected network to set self.accel.

            Your task below is formulate the input appropriately to provide
            to the fully connected network.

            Implementation Instructions:
            1a. Initialize model_inp as a list of length 3. (Done for you)
             b. The first value should be accel_amt (See above for desc. of accel_amt)
             c. The second value should be brake_amt. (See above for desc. of brake_amt)
             d. The third value should be the current velocity (No need for abs value)
            model_inp = [accel_amt, brake_amt, velocity]

            The ouptut is the predicted acceleration which should
            account for all internal dynamics.
            '''
            model_inp = [0, 0, 0]

            #CODE HERE (Delete exception too)
            raise Exception("You forgot to fill Complex Input Formulation in the Controller Model")

            self.accel = self.complex_accel_fcn.predict([model_inp])


        '''
        Acceleration is negated if in Reverse gear.
        '''
        if (self.gear == self.REVERSE):
            self.accel *= -1

        '''
        If the velocity is 0 make sure the car cannot accelerate backwards.
        This is to prevent moving backwards when braking in a stationary position.
        '''
        if (self.velocity == 0):
            if (self.gear == self.FORWARD):
                self.accel = max(self.accel, 0)
            elif (self.gear == self.REVERSE):
                self.accel = min(self.accel, 0)

        '''
        PART OF WEEK 2 HW

        Assuming constant velocity and acceleration over a small interval of length
        self.dt. Fill in the below update equations for position and velocity.

        HINT: position update should have a linear term in velocity, and a quadratic
              term in acceleration.
        '''
        '''
        UNCOMMENT AND FILL IN (Delete exception too)

        self.position +=
        self.velocity +=
        '''
        raise Exception("You forgot to fill in pos/vel dynamics in the Controller Model")

        #These ensure that the velocity is never against the current gear setting.
        if (self.gear == self.FORWARD):
            self.velocity = max(self.velocity, 0)
        elif (self.gear == self.REVERSE):
            self.velocity = min(self.velocity, 0)

        #Increment the internal time
        self.T += self.dt


    #Used for switching gear (None, the initial state inhibits movement)
    def set_gear(self, gear):
        if (gear != self.FORWARD and gear != self.REVERSE):
            raise Exception(f"Invalid gear provided {gear}")

        if (gear != self.gear):
            if (self.gear is None):
                self.gear = gear
            else:
                if (abs(self.velocity) < 0.01):
                    self.gear = gear
                else:
                    raise Exception(f"Speed must be below 0.01, current speed: {abs(self.velocity)}")

    #Since position is a relative measure you are free to zero at will
    def zero_position(self):
        self.position = 0

    #Steer to desired angle (-1 (right) to 1 (left))
    def steer_to(self, ang):
        self.steering_angle = max(-1, min(ang, 1))

    '''
    Crawls the working directory up to the top most folder for which
    car_iface should be a child directory.  Adds the name of the file
    to get the learned weights for the complex model.
    '''
    def complex_weights_fp(self):
        cur_dir = os.path.dirname(__file__)
        return os.join(cur_fp, "complex_accel")
