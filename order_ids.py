#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import random
import csv

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid.pkl", 'rb') as f:
    database = pickle.load(f)

pedestrian_per_set = 15
out_list = []
reserved_list = []
for i in range(0, 50):
    playlist = []
    for int_length in [1.0, 3.0, 5.0, 7.0, 9.0, 12.0]:
        ped_candidate = []
        for name, val in database.items():
            if val.get("results") is None or name.rsplit("_", 1)[0].endswith("tl"):
                continue
            if float(name.rsplit("_", 1)[-1]) == int_length and name.rsplit("_", 1)[0] not in reserved_list:
                ped_candidate.append(name)

        if len(ped_candidate) <= pedestrian_per_set:
            reserved_list += [id.rsplit("_", 1)[0] for id in ped_candidate]
            playlist += ped_candidate
        else:
            list = random.choices(ped_candidate, k=pedestrian_per_set)
            reserved_list += [id.rsplit("_", 1)[0] for id in list]
            playlist += list

    print(playlist)
    out_list.append(playlist)


with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)
