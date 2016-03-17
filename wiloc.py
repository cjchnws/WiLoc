#!/usr/bin/env python

import numpy as np
import cv2 as cv

from EKF import EKF
from wiloc_model import WiLocModel


param_sets = {
  # a first working parameter set with a simple constant model
  # and direct control
  'basic': {
    'A': np.mat([
      [1, 0],
      [0, 1],
      ]),
    'B': np.mat([
      [1, 0],
      [0, 1],
      ]).T,
    'R': np.array([0.1, 0.1]),
    # start off with a very bad estimate (of 100m variance)
    'Sigma_0': np.eye(2) * 100
  },

  # a const vel model (not working yet)
  'vel1': {
    'A': np.mat([
      [1, 0, 0.1, 0],
      [0, 1, 0, 0.1],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      ]),
    'B': np.mat([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      ]).T,
    'R': np.array([0.05, 0.05, .1, .1]),
    'Sigma_0': np.eye(2) * 100
  }


}


np.set_printoptions(precision=3, linewidth=200)

if __name__ == "__main__":
    locator = WiLocModel()
    # locator.load_data_from_json()
    # locator.save_pickle()
    # exit()
    locator.load_pickle()
    # locator.generate_simple()
    locator.compute_basics()
    # valid_points = locator.detect_motion()
    # locator.data = locator.data[:,valid_points]
    locator.compute_basics()
    locator.compute_transform()
    print 'LMS residual error=%.2f' % locator.compute_trans_error()
    print 'A=', locator.A
    # locator.print_histogram(locator.query(0, 0))
    # locator.print_histogram(locator.query(1, 0.1))
    # locator.print_histogram(locator.query(2, 0))
    # locator.print_histogram(locator.query(3, 0))

    # print 'JacH', locator.JacH(np.array([[0,0]]).T)
    # print 'JacH', locator.JacH(np.array([[1,0]]).T)
    # print 'JacH', locator.JacH(np.array([[2,0]]).T)
    # print 'JacH', locator.JacH(np.array([[3,0]]).T)
    # print 'funcH', locator.funcH(np.array([2,0]).T)

    ##################################################
    # initialise the Parameters
    param_set = 'basic'

    A = param_sets[param_set]['A']
    B = param_sets[param_set]['B']
    R = param_sets[param_set]['R']
    Sigma = param_sets[param_set]['Sigma_0']

    # get Q from the actually trained model
    Q = locator.C_W

    ##################################################
    # create EKF
    ekf = EKF(R, Q, A=A, B=B, funcH=locator.funcH, funcJacobianH=locator.JacH)

    # find those points where there was actual motion only...
    # Yes this will induce large leaps in between, which is why
    # a constant velocity model won't work, but the data is
    # just not there properly.
    valid_points = locator.detect_motion()

    img = np.zeros((600, 300, 3), np.uint8) + 100
    locator.display_wifi_maps(img)

    # start off at the wrong place (by 20m)
    mu = locator.data[0:2, valid_points[0]] - 20

    for p in valid_points:
        # image to display everything
        img = np.zeros((600, 300, 3), np.uint8) + 100
        # draw all valid locations in white
        locator.draw_locations(img, locator.data[0:2, valid_points],
                               color=(255, 255, 255))

        # get the next observation from the data
        z = locator.data[locator.split_PW:, p]

        # get the ground truth mu and display in green
        mu_true = locator.data[0:2, p]
        locator.draw_state(img, mu_true, None, color=(0, 255, 0))

        # get the "odometry" from the data
        u = locator.motion_vecs[0:2, p]

        print 'vel: ', locator.motion_vecs[:, p]
        print 'real mu: ', mu_true
        print 'real u: ', u
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
