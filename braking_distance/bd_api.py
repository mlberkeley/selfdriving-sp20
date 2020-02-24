import sys
import os
import numpy as np

sys.path.append('../')

from importlib import import_module
from utils import nn
import car_iface.controller_model as controller_model

'''
Braking Distance Estimator class.
Used for determining how much to depress the brake pedal
to stop at certain distance.
'''
class Braking_Distance_Estimator():
    def __init__(self, car_model, estimator_type):
        '''
        Allowable combinations of car_model and estimator_type are
        ("simple", "fcn")
        ("simple", "analytical")
        ("complex", "fcn")

        No analytical solution for complex internal model.
        '''
        if (estimator_type not in {"analytical", "fcn"}):
            raise Exception(f"Illegal estimator_type, f{estimator_type}, must be 'analytical' or 'fcn'")

        if (car_model not in {"simple", "complex"}):
            raise Exception(f"Illegal estimator_type, f{estimator_type}, must be 'simple' or 'complex'")

        if (estimator_type == 'analytical' and car_model == 'complex'):
            raise Exception("No analytical estimator for complex car model")

        self.estimator_type = estimator_type

        '''
        If using a fcn load parameters from Brakind Distance directory
        two inputs (intitial velocity, stopping distance), output must be between 0 and 1
        '''
        if (estimator_type == "fcn"):
            self.fcn = nn.fcn(model_name = os.path.join(self.bd_fp(), f"{car_model}_bd"), num_inputs = 2, out_range = (0, 1))

        '''
        If using analytical approach get coefficients from controller
        model (pedal weights, friction constant, and rolling bias)
        '''
        if (estimator_type == "analytical"):
            ci = controller_model.Car_Interface()
            self.brake_weight = ci.brake_weight
            self.rolling_bias = ci.rolling_bias
            self.friction_constant = ci.friction_constant

    '''
    Core utility: returns brake pedal depression as a function of
    current velocity and desired stopping distance using the appropriate
    estimator.
    '''
    def estimate(self, vel, stopping_distance):

        inp = [vel, stopping_distance]
        if (self.estimator_type == "analytical"):
            return self.simple_analytical_approx(inp)
        else:
            return self.fcn([inp]).detach().numpy()


    '''
    The next two functions are for the analytical method.

    The first calculates stopping distance as a function of
    the initial velocity and the constant brake pedal amount.
    This function is derived using differential equations.

    The second function uses applies binary search to the first
    function to efficiently calculate the brake amount that yields
    roughly the desired stopping distance given an initial velocity.
    '''
    def simple_analytical_sd(self, initial_velocity, amt):
        #CODE HERE: Paste corresponding code from Jupyter Notebook

    def simple_analytical_approx(self, inp, tol = 1e-5, min_amt = 0, max_amt = 1):
        #CODE HERE: Paste corresponding code from Jupyter Notebook

    #File path to Braking Distance folder
    def bd_fp(self):
        import settings

        cur_dir = os.path.dirname(__file__)
        internal_path = os.path.join(cur_dir, '../../internal_only/braking_distance/')
        if settings.use_internal_if_available and os.path.exists(internal_path):
            return internal_path
        return cur_dir
