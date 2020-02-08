import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilterToy:
    def __init__(self):
        self.v = 0
        self.prev_x = 0
        self.prev_t = 0
    def predict(self,t):
        prediction = self.v * (t - self.prev_t) + self.prev_x
        return prediction
    def measure_and_update(self,x,t):
        new_v = (x - self.prev_x) / (t - self.prev_t)
        self.v += 0.5 * (new_v - self.v)
        predicted_x = self.v * (t - self.prev_t) + self.prev_x
        self.prev_x = (x + predicted_x) / 2.
        self.prev_t = t
        return


sim_run(options,KalmanFilterToy)
