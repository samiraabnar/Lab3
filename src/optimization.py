import random
import time
import copy

''' class for Parameter Values'''
class parameter_value:

    def __init__(self, minim=0.0, maxim=1.0):
        self.clamped = False
        self.min = minim
        self.max = maxim
        self.value = self.min

    def update(self, new_value):
        if new_value >= self.min and new_value <= self.max:
            self.value = new_value


class Optimization:
    mRandom = random.Random(time.time())

    @staticmethod
    def greedy_search(self,model,data
                      #you may add extra parameters if you need
                     ):
        raise NotImplementedError




    @staticmethod
    def hill_climbing(maxIterations, max_jump, model, data
                      # you may add extra parameters if you need
                      ):
        raise NotImplementedError


    @staticmethod
    def get_hillClimbing_next_parameter(current_parameters,max_jump,changing_param_index):
        next_params = copy.deepcopy(current_parameters)

        var = min(next_params[changing_param_index].max - current_parameters[changing_param_index].value, current_parameters[changing_param_index].value - next_params[changing_param_index].min) * max_jump
        next_params[changing_param_index].update(Optimization.mRandom.gauss(current_parameters[changing_param_index].value, var))

        return next_params
