#!/usr/bin/env python

import numpy as np


data = np.mat([
  [1.0, 1.0, -40, -60, -80],
  [2.0, 1.0, -60, -40, -60],
  [2.0, 2.0, -80, -60, -40],
  [1.0, 2.0, -60, -80, -60],
  [3.0, 1.0, -80, -60, -70],
  ])

P = np.transpose(data[0:, 0:2])
P = np.vstack([P, [1, 1, 1, 1, 1]])
W = np.transpose(data[0:, 2:])

# A*P = W <=> A= W \ P

Pp = np.linalg.pinv(P)

print P

A = W * Pp

print A

print A * P
