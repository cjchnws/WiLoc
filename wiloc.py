#!/usr/bin/env python

import numpy as np
from pymongo import MongoClient
from json import load
from datetime import datetime
from math import sqrt
import time
import pickle

import cv2 as cv

from dateutil.parser import parse


np.set_printoptions(precision=3, linewidth=200)


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
        Hn = np.zeros((H.shape[0], H.shape[1]+2))
        Hn[:,:-2] = H
        Hn[:,2:] = H
        print H.shape, Hn.shape
        H = np.mat(Hn)

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


class WiLoc:

    def __init__(self):
        cv.namedWindow('location')
        self.img = np.zeros([300, 300, 3])

        self.data = None
        self.times = None
        self.split_PW = 6

    def db_to_nw(self, db):
        # see https://en.wikipedia.org/wiki/DBm
        # print db/10.0
        return (10.0 ** (db/10.0)) * 1e6

    def db_to_rel_distance(self, db):
        # see https://en.wikipedia.org/wiki/Free-space_path_loss
        mw = self.db_to_nw(db)
        return 0.01 / (sqrt(mw))

    def load_pickle(self, filename="save.p"):
        d = pickle.load(open(filename, "rb"))
        self.data = d['data']
        self.times = d['times']
        print 'loaded'

    def save_pickle(self, filename="save.p"):
        pickle.dump({'data': self.data, 'times': self.times},
                    open("save.p", "wb"))
        print 'saved'

    def load_data_from_json(self, filename='data-aaf.json', limit=None,
                            skip=0, observation_limit=500):
        with open(filename, 'r') as infile:
            j = load(infile)

        c = 0
        new_d = []
        for v in j['data']:
            c += 1
            if c < skip:
                continue
            new_d.append(v)
            if limit is not None:
                if c >= limit:
                    break

        initial = {}
        c_bssids = {}
        s_bssids = {}
        for b in j['bssids']:
            c_bssids[b] = 0
            s_bssids[b] = 0

        for v in new_d:
            c_bssids[v['bssid']] += 1
            s_bssids[v['bssid']] += self.db_to_nw(float(v['signal']))

        for b in j['bssids']:
            if c_bssids[b] >= observation_limit:
                initial[b] = s_bssids[b] / c_bssids[b]

        n_bssids = len(initial)
        n_points = len(j['data'])
        data = np.mat(np.zeros((n_points, n_bssids+self.split_PW)))
        times = np.mat(np.zeros((n_points, 1)))

        c = 0

        for v in new_d:
            bssid = v['bssid']
            if bssid in initial:
                initial[bssid] = self.db_to_nw(float(v['signal']))
                data[c, 0] = v['position']['x']
                data[c, 1] = v['position']['y']
                data[c, 2] = v['position']['x'] ** 2
                data[c, 3] = v['position']['y'] ** 2
                data[c, 4] = 1  # offset
                data[c, 5] = 1  # offset
                data[c, self.split_PW:] = initial.values()
                t = parse(v['_meta']['inserted_at'])
                times[c, 0] = time.mktime(t.timetuple()) \
                    * 1000.0 + t.microsecond / 1000.0
                print 'complete: % 3.2f%%' % (c *100.0 / len(new_d))
                c += 1

        data = data[1:c, :]
        times = times[1:c, :]
        data = data.T

        print data.shape
        self.data = data
        self.times = times

    def coord_to_pixel(self, img, mu):
        s = img.shape
        dx = self.max_pos - self.min_pos
        scale = [dx[0] / s[1], dx[1] / s[0]]
        scale = np.max(scale) * 2
        p = (int(s[1] / 2.0 + (mu[0]) / scale),
             int(s[0] / 2.0 + (mu[1]) / scale))

        if p[0] < 0:
            return None
        if p[0] >= s[1]:
            return None
        if p[1] < 0:
            return None
        if p[1] >= s[0]:
            return None
        return p

    def draw_state(self, img, mu, sigma=None, color=(0, 0, 255)):
        p = self.coord_to_pixel(img, mu)
        cv.circle(img, p, 3, color, -1)


    def draw_locations(self, img, data, color=(0, 0, 255)):
        c = 0
        for r in data.T:
            self.draw_state(img, (r[0, 0], r[0, 1]), sigma=None, color=color)
            c += 1

    def detect_motion(self):
        self.motion_vecs = self.data[0:2, 1:]-self.data[0:2, 0:-1]
        motions = np.linalg.norm(self.motion_vecs, axis=0)

        self.big_diff = (motions > 0.01).nonzero()
        return self.big_diff[0]
        # print self.n_points-self.small_diff.size, vels[self.small_diff]

    def compute_basics(self):
        self.n_bssids, self.n_points = self.data.shape
        # substract the 3 coords fields
        self.n_bssids -= self.split_PW

        print 'n_points=%d, n_bssids=%d' % (self.n_points, self.n_bssids)

        self.mu = np.mean(self.data, axis=1)
        self.data_c = self.data

        # for times to work we need the corresponding robot_pose times,
        # not the wifi times!

        # self.timediff = self.times[1:, 0]-self.times[0:-1, 0] + 0.0000001
        # vels = np.divide(np.linalg.norm(self.velocities, axis=0), np.array(self.timediff.T))

        # self.small_diff,b = (vels > 0.01).nonzero()
        # print self.small_diff.size, vels[0, self.small_diff], np.min(self.timediff)
        # print self.n_points-self.small_diff.size, vels[self.small_diff]
        self.min_pos = np.min(self.data[0:2, :], axis=1)
        self.max_pos = np.max(self.data[0:2, :], axis=1)
        self.min_dist = np.min(self.data[self.split_PW:, :], axis=1)
        self.max_dist = np.max(self.data[self.split_PW:, :], axis=1)
        print 'minimum pos', self.min_pos.T
        print 'maximum pos', self.max_pos.T
        print 'minimum dist', self.min_dist.T
        print 'maximum dist', self.max_dist.T


        #draw_locations(img, self.data_c[0:2,:],2)

        #cv.imshow('location', img)

        #cv.waitKey(0)
        self.C_full = (self.data_c * self.data_c.T) / self.n_points

        self.mu_P = self.mu[0:self.split_PW]
        self.C_P = self.C_full[0:self.split_PW, 0:self.split_PW]
        print "mu_p=", self.mu_P.T
        print "C_p=", self.C_P
        self.mu_W = self.mu[self.split_PW:]
        self.C_W = self.C_full[self.split_PW:, self.split_PW:]
        print "mu_w=", self.mu_W.T
        print "C_w=", np.diag(self.C_W)

    def compute_trans_error(self):
        D = np.array((self.A * self.P) - self.W)
        return np.sqrt(np.sum(D * D) / (self.n_points*self.n_bssids))

    def compute_transform(self):
        # matrix of all coordinates
        self.P = self.data_c[0:self.split_PW, :]
        # matrix of all wifi signals
        self.W = self.data_c[self.split_PW:, :]

        # find least-square solution A to transform form P => W
        # A*P = W <=> A= W \ P
        self.Pp = np.linalg.pinv(self.P)

        self.A = self.W * self.Pp

    def query(self, x, y):
        pos = np.mat([x, y]).T
        obs = self.funcH(pos)
        return obs

#     def update(self, mu, Sigma, z):
#         print 'Kalman update: '
#         obs = self.funcH(mu)
#         print '  exp obs=', obs
#         H = self.JacH(mu)
#         print "  mu=", mu
#         print "  Sigma=", Sigma
#         print '  H=', H
#         Q = np.eye(self.n_bssids) * 0.1
#         print '  Q=', Q
#         K = Sigma * H.T * np.linalg.inv(H * Sigma * H.T + Q)
#         print '  K=', K
#         mu_e = mu + K * (z - self.funcH(mu))
#         print '  me_e=', mu_e
#         Sigma_e = (np.eye(2) - K * H) * Sigma
#         print '  Sigma_e=', Sigma_e
# #        sigma_new = (np.eye())
#         return mu_e, Sigma_e

    def augment_state(self, orig):
        # assumes the original state to have 2 coordinates
        a = np.asarray(orig)
        ones = np.ones(a.shape)
        if self.split_PW < 4:
            return a
        else:
            return np.vstack((a, a * a, ones))

    def JacH(self, state):
        j = np.asmatrix(np.zeros((self.n_bssids, 2)))
        j[:,0] = self.A[:,2] * 2 * np.asmatrix(state[0]) + self.A[:,0]
        j[:,1] = self.A[:,3] * 2 * np.asmatrix(state[1]) + self.A[:,1]
        return j

    def funcH(self, state):
        if state.ndim == 1:
            state = state.reshape((state.size, 1))
        # first add the non-linear state descriptors
        state = self.augment_state(state[0:2,:])
        # zero-mean it
        query = np.asmatrix(state)
        # do the linear transform into observation space
        results = (self.A * query)
        return results

    def grid_query(self, xr, yr):
        coords = [[a, b] for a in xr for b in yr]
        results = self.funcH(np.mat(coords).T)
        results = np.reshape(np.asarray(results).T,
                             (len(xr), len(yr), self.n_bssids))
        return results

    def grid_query_OFF(self, xr, yr):
        results = np.zeros([len(xr), len(yr), self.n_bssids])
        for xi in range(0, len(xr)):
            for yi in range(0, len(yr)):
                state = np.mat([xr[xi], yr[yi]]).T
                expObs = self.funcH(state)
                results[xi, yi, :] = np.reshape(expObs, (self.n_bssids))
        return results

    def print_histogram(self, vec, res=1):
        for x in np.nditer(vec):
            bars = '*' * int(x / res)
            print '  % 4.2f %s' % (x, bars)
        print '==='

    def augment_raw_data(self, raw_data):
        squared = raw_data[0:2, :] ** 2
        ones = np.ones(squared.shape)
        data = np.mat(np.vstack((raw_data[0:2, :],
                                squared, ones, raw_data[2:, :])))
        self.times = np.mat([[0, 1, 2, 3]])
        return data

    def generate_simple(self):
        raw_data = np.array([
          [0, 0,    9,  6, 10],
          [1, 0.1, 10,  9,  9],
          [2, 0.1,  9, 10,  6],
          [3, 0,    6,  9,  1]
          ]).T
        self.data = self.augment_raw_data(raw_data)
        self.times = np.arange(0, self.data.shape[1])

    def display_wifi_maps(self, resolution=0.1):
        xb = self.min_pos[0, 0]
        xe = self.max_pos[0, 0]
        xsteps = int((xe - xb) / resolution)
        yb = self.min_pos[1, 0]
        ye = self.max_pos[1, 0]
        ysteps = int((ye - yb) / resolution)
        print ysteps
        r = locator.grid_query(np.linspace(xb, xe, xsteps),
                               np.linspace(yb, ye, ysteps))
        for i in range(0, self.n_bssids):
            print i
            img = cv.normalize((r[:, :, [i,i,i]]), alpha=0,
                               beta=1, norm_type=cv.NORM_MINMAX)
            cv.putText(img, str(i), (ysteps/2, xsteps/2), cv.FONT_HERSHEY_PLAIN,
                       2, (0, 0, 255))

            cv.imshow('location', img)
            key = cv.waitKey(0)
            if key == ord('q'):
                break


if __name__ == "__main__":
    locator = WiLoc()
    # locator.load_data_from_json()
    # locator.save_pickle()
    # exit()
    locator.load_pickle()
    #locator.generate_simple()
    locator.compute_basics()
    # valid_points = locator.detect_motion()
    # locator.data = locator.data[:,valid_points]
    locator.compute_basics()
    locator.compute_transform()
    print 'LMS residual error=%.2f' % locator.compute_trans_error()
    print 'A=', locator.A
    locator.print_histogram(locator.query(0, 0))
    locator.print_histogram(locator.query(1, 0.1))
    locator.print_histogram(locator.query(2, 0))
    locator.print_histogram(locator.query(3, 0))

    # print 'JacH', locator.JacH(np.array([[0,0]]).T)
    # print 'JacH', locator.JacH(np.array([[1,0]]).T)
    # print 'JacH', locator.JacH(np.array([[2,0]]).T)
    # print 'JacH', locator.JacH(np.array([[3,0]]).T)
    # print 'funcH', locator.funcH(np.array([2,0]).T)

    A = np.mat([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      ])
    B = np.mat([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      ]).T
    R = np.array([0.05, 0.05, .1, .1])
    Q = np.eye(locator.n_bssids) * .1

    ekf = EKF(R, Q, A=A, B=B, funcH=locator.funcH, funcJacobianH=locator.JacH)

    print locator.data[locator.split_PW:,0]
    #z = np.mat([6, 9, 1]).T
    

    img  = np.zeros((600,300,3), np.uint8)+100

    valid_points = locator.detect_motion()

    locator.draw_locations(img, locator.data[0:2,valid_points], color=(255, 255, 255))

    Sigma = np.eye(4) * 10
    mu = locator.data[0:4,valid_points[0]] - 20
    mu[2:4,0] = [[0,0]]
    for p in valid_points:
        img  = np.zeros((600,300,3), np.uint8)+100
        z = locator.data[locator.split_PW:,p]
        mu_true = locator.data[0:2,p]
        print 'vel: ', locator.motion_vecs[:,p]
        print 'real mu: ', mu_true
        u = locator.motion_vecs[0:2,p]
        print 'real u: ', u
        locator.draw_state(img, mu_true, None, color=(0, 255, 0))
        mu_p, Sigma_p = ekf.predict(mu, Sigma, u=u)
        locator.draw_state(img, mu_p, Sigma_p, color=(0, 255, 255))
        mu, Sigma = ekf.update(mu_p, Sigma_p, z)
        locator.draw_state(img, mu, Sigma, color=(0, 0, 255))
        cv.imshow('location', img)
        key = cv.waitKey(10000)
        if key == ord('q'):
            break



    # print 'JacH', locator.JacH(np.array([[0,-30]]).T)
    # print 'JacH', locator.JacH(np.array([[0,0]]).T)
    # print 'JacH', locator.JacH(np.array([[0,30]]).T)
    # print 'funcH', locator.funcH(np.array([0,0]).T)

    #locator.display_wifi_maps()
    # locator.print_histogram(locator.query(0, -70))
    # locator.print_histogram(locator.query(0, 0))
    # locator.print_histogram(locator.query(0, 50))
    # locator.print_histogram(locator.query(-34, 50))
    # locator.print_histogram(locator.query(-34, -70))
