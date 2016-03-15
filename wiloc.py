#!/usr/bin/env python

import numpy as np
from pymongo import MongoClient
from json import load
from datetime import datetime

import cv2 as cv

from dateutil.parser import parse

def draw_location(img, position, scale=10):
    s = img.shape
    p = (int(s[0] / 2.0 + position[0] * scale),
         int(s[1] / 2.0 + position[1] * scale))
    #print p
    if p[0] < 0:
        return
    if p[0] >= s[0]:
        return
    if p[1] < 0:
        return
    if p[1] >= s[1]:
        return

    cv.circle(img, p, 3, (0, 0, 255), -1)

def draw_locations(img, data, scale=10):
    c = 0
    minim = np.min(data, axis=1)
    maxim = np.max(data, axis=1)
    print minim, maxim
    for r in data.T:
        draw_location(img, (r[0,0], r[0,1]), scale)
        c += 1
    print "drew circles: ", c


cv.namedWindow('location')


img = np.zeros([300, 300, 3])


#data = np.mat([[0, 0],[10,-10],[10,0]]).T

#draw_locations(img,data)

#cv.imshow('location', img)

#cv.waitKey(1)


with open('data-aaf.json', 'r') as infile:
    j = load(infile)

c = 0





new_d = {}
for k in j['data']:
    c += 1
    #if c<10000:
    #    continue
    v = j['data'][k]
    nk = parse(k)
    new_d[nk] = v
    #if c > 20000:
    #    break
    #print nk

keys = new_d.keys()
keys.sort()
#print keys



last = {}
c_bssids = {}
s_bssids = {}
for b in j['bssids']:
    c_bssids[b] = 0
    s_bssids[b] = 0


for k in keys:
    v = new_d[k]
    c_bssids[v['bssid']] += 1
    s_bssids[v['bssid']] += float(v['quality'])


for b in j['bssids']:
    if c_bssids[b] > 500:
        last[b] = s_bssids[b] / c_bssids[b]

n_bssids = len(last)
n_points = len(keys)
data = np.mat(np.zeros((n_points, n_bssids+3)))

c = 0
for k in keys:
    v = new_d[k]
    bssid = v['bssid']
    if bssid in last:
        last[bssid] = v['quality']
        data[c, 0] = v['position']['x']
        data[c, 1] = v['position']['y']
        data[c, 2] = 1
        data[c, 3:] = last.values()
        c += 1

data = data[1:c, :]

data = data.T

print data.shape


mu = np.mean(data, axis=1)
data_c = data-mu

draw_locations(img, data_c[0:2,:],2)

cv.imshow('location', img)

cv.waitKey(0)
#C_full = (data_c*data_c.T) / n_points

#C_P = C_full[0:2,0:2]
#C_W = C_full[3:,3:]
#C_PW = C_full[0:2,3:]
#C_WP = C_full[3:,0:2]
mu_P = mu[0:2]
mu_W = mu[3:]

#C_PgW = C_P - C_PW * np.linalg.pinv(C_W) * C_WP

#print C_PgW

P = data_c[0:2, :]
W = data_c[3:, :]
# A*P = W <=> A= W \ P

Pp = np.linalg.pinv(P)

A = W * Pp

query = np.mat([100, 100]).T - mu_P

R = (A * query) + mu_W

D = np.array((A * P) - W)
print R.shape
print R
print np.sqrt(np.sum(D * D) / (n_points*n_bssids))



cv.waitKey(0)
