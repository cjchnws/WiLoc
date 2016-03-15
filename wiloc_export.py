#!/usr/bin/env python

import numpy as np
from pymongo import MongoClient
from json import dump
from datetime import datetime


def read_mongo_wifi():
    mongo = MongoClient(host="localhost", port=52345)

    roslog = mongo.roslog

    cursor = roslog.wifiscanner.find()

    bssids = set()
    dataset = {}
    count = 0

    for e in cursor:
        values = e['status'][0]['values']

        d = {'_meta': e['_meta']}
        rp = roslog.robot_pose.find({'_meta.inserted_at': {
          '$gte': d['_meta']['inserted_at']}}).limit(1)
        d['position'] = rp[0]['position']
        d['_meta']['inserted_at'] = d['_meta']['inserted_at'].isoformat()

        for v in values:
            d[v['key']] = v['value']
        bssids.add(d['bssid'])

        count += 1
        dataset[d['_meta']['inserted_at']] = d

        print count
#        if count > 10:
#          break

    print bssids
    #print dataset

    j = {
      'bssids': list(bssids),
      'data': dataset
    }

    with open('data.json', 'w') as outfile:
        dump(j, outfile, sort_keys=True, indent=4)

read_mongo_wifi()

