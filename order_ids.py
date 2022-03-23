#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import random
import csv
import pandas as pd



with open("./data/PIE_data/experiment/database.pkl", 'rb') as f:
    database = pickle.load(f)

pedestrian_per_set = 10
out_list = []
int_length_list = [1.0, 3.0, 5.0, 8.0]
# state: 0=nocross/green 1=cross/red, 8.0:0=avairable 1=arrocated -1=not avairable
pd_reserve_list = pd.DataFrame(columns=["id", "state", "length", "allocate"])
tl_reserve_list = pd.DataFrame(columns=["id", "state", "length", "allocate"])
for name, val in database.items():
    if val.get("label") == "pedestrian":
        if val.get("id") not in list(pd_reserve_list.id):
            for length in int_length_list:
                buf = pd.DataFrame([(val.get("id"), val.get("crossing"), float(length), -1)], columns=pd_reserve_list.columns)
                pd_reserve_list = pd.concat([pd_reserve_list, buf], ignore_index=True)

        pd_reserve_list.loc[(pd_reserve_list.id == val.get("id")) & (pd_reserve_list.length == val.get("int_length")),"allocate"] = 0

    elif val.get("label") == "traffic_light":
        if val.get("id") not in list(tl_reserve_list.id):
            for length in int_length_list:
                buf = pd.DataFrame([(val.get("id"), val.get("state"), float(length), -1)], columns=tl_reserve_list.columns)
                tl_reserve_list = pd.concat([tl_reserve_list, buf], ignore_index=True)

        tl_reserve_list.loc[(tl_reserve_list.id == val.get("id")) & (tl_reserve_list.length == val.get("int_length")),"allocate"] = 0

extractable_flag = True
for i in range(1, 20):
    playlist = []
    extracted_list = []
    for int_length in int_length_list:
        for state in ["0", "1"]: # state
            candidate_list = pd_reserve_list[(pd_reserve_list.length == int_length) & (pd_reserve_list.state == state) & (pd_reserve_list.allocate == 0) & ~pd_reserve_list["id"].isin(extracted_list)]

            if len(candidate_list) < 5:
                print("reset allocation for pedestrian index:{}".format(i))
                pd_reserve_list.loc[(pd_reserve_list.length == int_length) & (pd_reserve_list.state == state), "allocate"] = 0
                candidate_list = pd_reserve_list[(pd_reserve_list.length == int_length) & (pd_reserve_list.state == state) & (pd_reserve_list.allocate == 0) & ~pd_reserve_list["id"].isin(extracted_list)]

            for index, candidate in candidate_list.sample(n=5).iterrows():
                extracted_list.append(str(candidate.id))
                pd_reserve_list.loc[(pd_reserve_list.id == candidate.get("id")) & (pd_reserve_list.length == int_length), "allocate"] = 1
                playlist.append(candidate.id + "_" + str(int_length))

    out_list.append(playlist)
    playlist = []

    for int_length in int_length_list:
        for state in ["0", "1"]: # state
            candidate_list = pd_reserve_list[(pd_reserve_list.length == int_length) & (pd_reserve_list.state == state) & (pd_reserve_list.allocate == 0) & ~pd_reserve_list["id"].isin(extracted_list)]

            if len(candidate_list) < 5:
                print("reset allocation for pedestrian index:{}".format(i))
                pd_reserve_list.loc[(pd_reserve_list.allocate != -1), "allocate"] = 0
                candidate_list = pd_reserve_list[(pd_reserve_list.length == int_length) & (pd_reserve_list.state == state) & (pd_reserve_list.allocate == 0) & ~pd_reserve_list["id"].isin(extracted_list)]

            for index, candidate in candidate_list.sample(n=5).iterrows():
                extracted_list.append(str(candidate.id))
                pd_reserve_list.loc[(pd_reserve_list.id == candidate.get("id")) & (pd_reserve_list.length == int_length), "allocate"] = 1
                playlist.append(candidate.id + "_" + str(int_length))

    out_list.append(playlist)
    playlist = []

    for int_length in int_length_list:
        for state in [0, 1]: # state
            candidate_list = tl_reserve_list[(tl_reserve_list.length == int_length) & (tl_reserve_list.state == state) & (tl_reserve_list.allocate == 0) & ~tl_reserve_list["id"].isin(extracted_list)]
            if len(candidate_list) < 5:
                print("reset allocation for traffic light index:{}".format(i))
                tl_reserve_list.loc[(tl_reserve_list.allocate != -1), "allocate"] = 0
                candidate_list = tl_reserve_list[(tl_reserve_list.length == int_length) & (tl_reserve_list.state == state) & (tl_reserve_list.allocate == 0) & ~tl_reserve_list["id"].isin(extracted_list)]

            for index, candidate in candidate_list.sample(n=5).iterrows():
                extracted_list.append(str(candidate.id))
                tl_reserve_list.loc[(tl_reserve_list.id == candidate.get("id")) & (tl_reserve_list.length == int_length),"allocate"] = 1
                playlist.append(candidate.id + "_" + str(int_length))

    out_list.append(playlist)


with open("./data/PIE_data/experiment/playlist.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)
