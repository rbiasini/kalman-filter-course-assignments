import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = True
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = True
options['RECIEVE_INPUTS'] = True

class KalmanFilter:
    def __init__(self):
        # Initial State x, y, vx, vy
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[100., 0., 0., 0., 0.],
                            [0., 100., 0., 0., 0.],
                            [0., 0., 100., 0., 0.],
                            [0., 0., 0., 100., 0.],
                            [0., 0., 0., 0., 100.]])

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

        # Measurement Function
        self.H = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[5., 0., 0.],
                            [0., 5., 0.],
                            [0., 0., 1.]])

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

    def recieve_inputs(self, u_steer, u_pedal):
        return

sim_run(options,KalmanFilter)
