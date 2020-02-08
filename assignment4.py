import numpy as np
from sim.sim2d_prediction import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['ALLOW_SPEEDING'] = True

class KalmanFilter:
    def __init__(self):
        # Initial State x, y, vx, vy
        self.x = np.matrix([[55.],
                            [3.],
                            [5.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]])

        # Uncertainity Matrix
        self.Q = np.matrix([[0.1, 0., 0., 0., 0.],
                            [0., 0.1, 0., 0., 0.],
                            [0., 0., 0.1, 0., 0.],
                            [0., 0., 0., 0.1, 0.],
                            [0., 0., 0., 0., 0.1]])

        # Next State Function
        self.F = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])

        # Input
        self.B = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]])

        # Measurement Function
        self.H = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[5.0, 0.],
                            [0., 5.0]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])


    def predict(self, dt):
        # Put dt into the state transition matrix.
        self.F[3,4] = dt
        self.F[0,2] = np.cos(self.x[3]) * dt
        self.F[1,2] = np.sin(self.x[3]) * dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F) + self.Q
        print(self.x)
        return [self.x[0], self.x[1]]

    def measure_and_update(self,measurements,dt):
        self.F[3,4] = dt
        self.F[0,2] = np.cos(self.x[3]) * dt
        self.F[1,2] = np.sin(self.x[3]) * dt
        Z = np.matrix(measurements[0:3])
        y = np.transpose(Z) - self.H * self.x
        S = self.H * self.P * np.transpose(self.H) + self.R
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P
        self.v = self.x[2,0]
        return [self.x[0], self.x[1]]

    def predict_red_light(self,light_location):
        light_duration = 3
        F_new = np.copy(self.F)
        F_new[3,4] = light_duration
        F_new[0,2] = np.cos(self.x[3]) * light_duration
        F_new[1,2] = np.sin(self.x[3]) * light_duration
        x_new = F_new * self.x
        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

    def predict_red_light_speed(self, light_location):
        light_duration = 3
        F_new = np.copy(self.F)
        F_new[3,4] = light_duration
        F_new[0,2] = np.cos(self.x[3]) * light_duration
        F_new[1,2] = np.sin(self.x[3]) * light_duration
        x_new = F_new * self.x + (light_duration**2) / 2 * 1.5
        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

for i in range(0,5):
    sim_run(options,KalmanFilter,i)
