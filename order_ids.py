#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import ramdom
import csv

with open("/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
    database = pickle.load(f)


reserved_list = []
out_list = []
for i in range(0, 20):
    playlist = []
    for int_length in [0.0, 1.0, 3.0, 5.0, 7.0 ,9.0]
        candidate = [name for name in database.keys() if (float(name.split("_", 1)) == int_length and name.split("_", 0) not in reserved_list)]
        list = ramdom.choises(candidate, 10)
        reserved_list += [id.splist("_", 0) for id in list]
        playlist += list

    out_list.append(play_list)


with open("/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/playlist.csv", 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)
