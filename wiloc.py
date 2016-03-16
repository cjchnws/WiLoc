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

class WiLoc:

    def __init__(self):
        cv.namedWindow('location')
        self.img = np.zeros([300, 300, 3])

        self.data = None
        self.times = None
        self.split_PW = 4


    def db_to_nw(self, db):
        # see https://en.wikipedia.org/wiki/DBm
        # print db/10.0
        return (10.0 ** (db/10.0)) * 1e6

    def db_to_rel_distance(self, db):
        # see https://en.wikipedia.org/wiki/Free-space_path_loss
        mw = self.db_to_nw(db)
        return 0.01 / (sqrt(mw))

    #print db_to_rel_distance(-40.0)
    #print db_to_rel_distance(-50.0)
    #print db_to_rel_distance(-60.0)
    #print db_to_rel_distance(-70.0)
    #print db_to_rel_distance(-80.0)

    #exit()

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
                initial[bssid] = self.db_to_nw(float(v['signal'])) # v['quality']
                data[c, 0] = v['position']['x']
                data[c, 1] = v['position']['y']
                data[c, 2] = v['position']['x'] ** 2
                data[c, 3] = v['position']['y'] ** 2
                data[c, self.split_PW:] = initial.values()
                t = parse(v['_meta']['inserted_at'])
                times[c, 0] = time.mktime(t.timetuple()) \
                    * 1000.0 + t.microsecond / 1000.0
                c += 1

        data = data[1:c, :]
        times = times[1:c, :]
        data = data.T

        print data.shape
        self.data = data
        self.times = times

    def draw_location(self, img, position, scale=10):
        s = img.shape
        p = (int(s[0] / 2.0 + position[0] * scale),
             int(s[1] / 2.0 + position[1] * scale))

        if p[0] < 0:
            return
        if p[0] >= s[0]:
            return
        if p[1] < 0:
            return
        if p[1] >= s[1]:
            return

        cv.circle(img, p, 3, (0, 0, 255), -1)

    def draw_locations(self, img, data, scale=10):
        c = 0
        for r in data.T:
            self.draw_location(img, (r[0, 0], r[0, 1]), scale)
            c += 1
        print "drew circles: ", c

    def compute_basics(self):
        self.n_bssids, self.n_points = self.data.shape
        # substract the 3 coords fields
        self.n_bssids -= self.split_PW

        print 'n_points=%d, n_bssids=%d' % (self.n_points, self.n_bssids)

        self.mu = np.mean(self.data, axis=1)
        self.data_c = self.data - self.mu
        self.velocities = self.data[0:2, 1:]-self.data[0:2, 0:-1]

        # for times to work we need the corresponding robot_pose times,
        # not the wifi times!

        # self.timediff = self.times[1:, 0]-self.times[0:-1, 0] + 0.0000001
        # vels = np.divide(np.linalg.norm(self.velocities, axis=0), np.array(self.timediff.T))

        # self.small_diff,b = (vels > 0.01).nonzero()
        # print self.small_diff.size, vels[0, self.small_diff], np.min(self.timediff)
        # print self.n_points-self.small_diff.size, vels[self.small_diff]
        self.min_pos = np.min(self.data[0:2, :], axis=1)
        self.max_pos = np.max(self.data[0:2, :], axis=1)
        self.min_dist = np.min(self.data[2:, :], axis=1)
        self.max_dist = np.max(self.data[2:, :], axis=1)
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
        pos = np.mat([x, y, x ** 2, y ** 2]).T
        query = pos - self.mu_P
        return (self.A * query) + self.mu_W

    def augment_state(self, orig):
        # assumes the original state to have 2 coordinates
        a = np.asarray(orig)
        return np.hstack((a, a * a))

    def grid_query(self, xr, yr):
        coords = [[a, b] for a in xr for b in yr]
        states = self.augment_state(np.mat(coords)).T
        query = np.asmatrix(states) #- self.mu_P
        results = (self.A * query) + self.mu_W
        #results = (np.eye(4) * query)
        results = np.reshape(np.asarray(results).T, (len(xr), len(yr), self.n_bssids))
        return results
        #print results[:,:,1]
        #print results[:,:,2]

    def print_histogram(self, vec, res=1):
        for x in np.nditer(vec):
            bars = '*' * int(x / res)
            print '  % 4.2f %s' % (x, bars)
        print '==='

        #cv.waitKey(0)

    def generate_simple(self):
        self.data = np.mat([
          [0, 0, 1, 1],
          [1, 1, 1, 0],
          [2, 4, 1, 1],
          [3, 9, 1, 4]
          ]).T
        self.times = np.mat([[0, 1, 2]])

    def display_wifi_maps(self, resolution=0.1):
        xb = self.min_pos[0,0]
        xe = self.max_pos[0,0]
        xsteps = int((xe - xb) / resolution)
        yb = self.min_pos[1,0]
        ye = self.max_pos[1,0]
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
    locator.load_pickle()
    # locator.generate_simple()
    locator.compute_basics()
    locator.compute_transform()
    print 'LMS residual error=%.2f' % locator.compute_trans_error()
    print 'A=', locator.A
    # locator.print_histogram(locator.query([0, 0]))
    # locator.print_histogram(locator.query([1, 1]))
    # locator.print_histogram(locator.query([2, 4]))
    # locator.print_histogram(locator.query([3, 9]))

    locator.display_wifi_maps()
    # locator.print_histogram(locator.query(0, -70))
    # locator.print_histogram(locator.query(0, 0))
    # locator.print_histogram(locator.query(0, 50))
    # locator.print_histogram(locator.query(-34, 50))
    # locator.print_histogram(locator.query(-34, -70))
