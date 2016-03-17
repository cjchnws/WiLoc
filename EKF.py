
import numpy as np


class EKF:

    def __init__(self, R, Q,
                 A=None, B=None, C=None,
                 funcH=None, funcJacobianH=None,
                 verbose=True):
        self.verbose = verbose
        if Q.ndim > 1:
            self.Q = Q
        else:
            self.Q = np.diag(Q)
        if R.ndim > 1:
            self.R = R
        else:
            self.R = np.diag(R)

        self.n_states = self.R.shape[0]
        self.n_obs = self.Q.shape[0]

        self.funcH = funcH
        self.funcJacobianH = funcJacobianH

        if A is not None:
            self.A = A
        else:
            self.A = np.eye(self.n_states)

        if B is not None:
            self.B = B
        else:
            self.B = np.eye(self.n_states)

        if C is not None:
            self.C = C
        else:
            self.C = np.ones((self.n_obs, self.n_states))

    def predict(self, mu, Sigma, u=None, A=None, B=None):
        if A is None:
            A = self.A
        if B is None:
            B = self.B

        if u is None:
            mu_p = A * mu
        else:
            mu_p = A * mu + B * u
        Sigma_p = A * Sigma * A.T + self.R

        return mu_p, Sigma_p

    def update(self, mu, Sigma, z):
        # giving it a shorter name:
        Q = self.Q
        # use the non-linear function to compute the expected outcome
        expect_obs = self.funcH(mu)

        # use the Jacobian to estimate the local (linear) changes
        H = self.funcJacobianH(mu)
        # Hn = np.zeros((H.shape[0], H.shape[1]+2))
        # Hn[:,:-2] = H
        # Hn[:,2:] = H
        # print H.shape, Hn.shape
        # H = np.mat(Hn)

        # compute the Kalman gain
        K = Sigma * H.T * np.linalg.inv(H * Sigma * H.T + Q)

        # estimate the new mean value after observation
        mu_e = mu + K * (z - expect_obs)
        # estimate the new Covariance after observation
        Sigma_e = (np.eye(self.n_states) - K * H) * Sigma

        if self.verbose:
            print 'Kalman update: '
            print '  expect_obs=', expect_obs
            print "  mu=", mu
            print "  Sigma=", Sigma
            print '  H=', H
            print '  K=', K
            print '  me_e=', mu_e
            print '  Sigma_e=', Sigma_e

        return mu_e, Sigma_e

