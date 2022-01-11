#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import random
import csv

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
    database = pickle.load(f)

tr_count = 0
ped_count = 0
for name in database.keys():
    id = name.rsplit("_", 1)[0]
    if id.endswith("tl"):
        tr_count += 1
    else:
        ped_count += 1

rate = ped_count / (tr_count + ped_count)
print("ped rate : {}".format(rate))

out_list = []
for i in range(0, 20):
    reserved_list = []
    playlist = []
    for int_length in [0.0, 1.0, 3.0, 5.0, 7.0 ,9.0]:
        ped_candidate = []
        tl_candidate = []
        for name, val in database.items():
            if val.get("results") is None:
                continue
            if float(name.rsplit("_", 1)[-1]) == int_length and name.rsplit("_", 1)[0] not in reserved_list:
                if name.rsplit("_", 1)[0].endswith("tl"):
                    tl_candidate.append(name)
                else:
                    ped_candidate.append(name)

        print("ped {} num:, {}".format(int_length, len(ped_candidate)), 30*rate)

        if len(ped_candidate) <= 20*rate:
            reserved_list += [id.rsplit("_", 1)[0] for id in ped_candidate]
            playlist += ped_candidate
        else:
            list = random.choices(ped_candidate, k=int(20*rate))
            reserved_list += [id.rsplit("_", 1)[0] for id in list]
            playlist += list

        if len(tl_candidate) <= 20*(1.0 - rate):
            reserved_list += [id.rsplit("_", 1)[0] for id in tl_candidate]
            playlist += tl_candidate
        else:
            list = random.choices(tl_candidate, k=int(20*(1.0-rate)))
            reserved_list += [id.rsplit("_", 1)[0] for id in list]
            playlist += list

    print(playlist)
    out_list.append(playlist)


with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)
