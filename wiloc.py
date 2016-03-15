#!/usr/bin/env python

import numpy as np
from pymongo import MongoClient
from json import load
from datetime import datetime


from dateutil.parser import parse

with open('data-aaf.json', 'r') as infile:
    j = load(infile)

c = 0

new_d = {}
for k in j['data']:
    v = j['data'][k]
    nk = parse(k)
    new_d[nk] = v
    c += 1
    if c > 20000:
        break
    #print nk

keys = new_d.keys()
keys.sort()
print keys


n_bssids = len(j['bssids'])
n_points = len(keys)
data = np.mat(np.zeros((n_points, n_bssids+3)))

last = {}
for b in j['bssids']:
    last[b] = -80


c = 0
for k in keys:
    v = new_d[k]
    bssid = v['bssid']
    last[bssid] = v['signal']
    data[c, 0] = v['position']['x']
    data[c, 1] = v['position']['y']
    data[c, 2] = 1
    data[c, 3:] = last.values()
    c += 1

print data.shape

P = np.transpose(data[0:, 0:3])
W = np.transpose(data[0:, 3:])

# A*P = W <=> A= W \ P

Pp = np.linalg.pinv(P)

A = W * Pp

print (A * P) - W

