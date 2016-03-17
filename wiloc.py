#!/usr/bin/env python

import numpy as np
import cv2 as cv

from EKF import EKF
from wiloc_model import WiLocModel


np.set_printoptions(precision=3, linewidth=200)

if __name__ == "__main__":
    locator = WiLocModel()
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
      [1, 0],
      [0, 1],
      ])
    B = np.mat([
      [1, 0],
      [0, 1],
      ]).T
    R = np.array([0.1, 0.1])
    Q = locator.C_W 
    # A = np.mat([
    #   [1, 0, 0, 0],
    #   [0, 1, 0, 0],
    #   [0, 0, 1, 0],
    #   [0, 0, 0, 1],
    #   ])
    # B = np.mat([
    #   [1, 0, 0, 0],
    #   [0, 1, 0, 0],
    #   ]).T
    # R = np.array([0.05, 0.05, .1, .1])
    # Q = locator.C_W

    ekf = EKF(R, Q, A=A, B=B, funcH=locator.funcH, funcJacobianH=locator.JacH)

    print locator.data[locator.split_PW:, 0]
    # z = np.mat([6, 9, 1]).T


    img  = np.zeros((600,300,3), np.uint8)+100

    valid_points = locator.detect_motion()

    locator.draw_locations(img, locator.data[0:2, valid_points],
                           color=(255, 255, 255))

    #Sigma = np.eye(4) * 10
    Sigma = np.eye(2) * 10
    mu = locator.data[0:2, valid_points[0]] - 20
    #mu[2:4,0] = [[0,0]]
    for p in valid_points:
        img  = np.zeros((600, 300, 3), np.uint8)+100
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
