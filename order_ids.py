#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import random
import csv
import pandas as pd



with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid.pkl", 'rb') as f:
    database = pickle.load(f)

pedestrian_per_set = 10
out_list = []
# state: 0=nocross/green 1=cross/red, 8.0:0=avairable 1=arrocated -1=not avairable
pd_reserved_list = pd.DataFrame(columns=["state", "8.0", "5.0", "3.0", "1.0"])
tl_reserved_list = pd.DataFrame(columns=["state", "8.0", "5.0", "3.0", "1.0"])

for name, val in database.items():
    if val.get("label") == "pedestrian":
        if val.get("id") in pd_reserved_list.id:
            buf = pd.DataFrame([val.get("crossing"), -1, -1, -1, -1], columns=pd_reserved_list.columns, index=val.get("id"))
            pd_reserved_list = pd.concat(pd_reserved_list, buf)

        pd_reserved_list.at[val.get("id"), val.get("int_length")] = 0

    elif val.get("label") == "traffic_light":
        if val.get("id") in tl_reserved_list.id:
            buf = pd.DataFrame([val.get("crossing"), -1, -1, -1, -1], columns=tl_reserved_list.columns, index=val.get("id"))
            tl_reserved_list = pd.concat(tl_reserved_list, buf)

            tl_reserved_list.at[val.get("id"), val.get("int_length")] = 0


extractable_flag = True
while extractable_frag:
    playlist = []
    extracted_list = []
    for int_length in ["8.0", "5.0", "3.0", "1.0"]:
        for state in [0, 1] # state
            candidate_list = pd_reserved_list[(pd_reserved_list[int_length] == 0) & (pd_reserved_list.state == state)]
            for candidate in candidate_list.sample(n=5):
                extracted_list.append(candidate.index)
                pd_reserved_list.at[candidate.index, int_length] == 1
                playlist.append(candidate.index + "_" + int_length)

    for int_length in ["8.0", "5.0", "3.0", "1.0"]:
        for state in [0, 1] # state
            candidate_list = pd_reserved_list[(pd_reserved_list[int_length] == 0) & (pd_reserved_list.state == state)].drop(extracted_list)
            for candidate in candidate_list.sample(n=5):
                extracted_list.append(candidate.index)
                pd_reserved_list.at[candidate.index, int_length] == 1
                playlist.append(candidate.index + "_" + int_length)

    for int_length in ["8.0", "5.0", "3.0", "1.0"]:
        for state in [0, 1] # state
            candidate_list = tl_reserved_list[(tl_reserved_list[int_length] == 0) & (tl_reserved_list.state == state)]
            for candidate in candidate_list.sample(n=5):
                tl_reserved_list.at[candidate.index, int_length] == 1
                playlist.append(candidate.index + "_" + int_length)

    print(playlist)
    out_list.append(playlist)


with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist_suzuki.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)
