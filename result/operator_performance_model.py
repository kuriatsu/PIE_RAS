#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os


def get_attrib(tree, id):
    for pedestrian in tree.iter("pedestrian"):
        if pedestrian.get("id") == id:
            return pedestrian.attrib.get("crossing_point"), pedestrian.attrib.get("num_lanes"), pedestrian.attrib.get("signalized")

    return None, None, None

def get_distance(tree, start_frame, end_frame):
    accum_dist = 0.0
    print(start_frame, end_frame)
    for frame in tree[start_frame : end_frame]:
        accum_dist += float(frame.attrib.get("OBD_speed"))  / (30 * 3.6)

    return accum_dist

sns.set(context='paper', style='whitegrid')
hue_order = ["traffic light", "crossing intention", "trajectory"]
eps=0.01
tl_black_list = [
"3_3_96tl",
"3_3_102tl",
"3_4_107tl",
"3_4_108tl",
"3_5_112tl",
"3_5_113tl",
"3_5_116tl",
"3_5_117tl",
"3_5_118tl",
"3_5_119tl",
"3_5_122tl",
"3_5_123tl",
"3_5_126tl",
"3_5_127tl",
"3_6_128tl",
"3_6_137tl",
"3_7_142tl",
"3_8_153tl",
"3_8_160tl",
"3_9_173tl",
"3_9_174tl",
"3_9_179tl",
"3_10_185tl",
"3_10_188tl",
"3_11_205tl",
"3_12_218tl",
"3_12_221tl",
"3_15_241tl",
"3_16_256tl",
"3_16_257tl",
]
opposite_anno_list = ["3_16_259tl", "3_16_258tl", "3_16_249tl"]

log_data = None
data_path = "/home/kuriatsu/Documents/PIE/log202203"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    filename =file.split("/")[-1]
    print("file:{}".format(filename))
    buf = pd.read_csv(file)
    count = int(filename.replace("log_data_", "").split("_")[-1].replace(".csv", ""))

    if count in [0, 1, 2]:
        print("skipped")
        continue

    buf["trial"] = filename.split("_")[-1].replace(".csv", "")
    buf["subject"] = filename.replace("log_data_", "").split("_")[0]
    buf["task"] = filename.replace("log_data_", "").split("_")[1]
    
    if log_data is None:
        log_data = buf
    else:
        log_data = log_data.append(buf, ignore_index=True)


vehicle_annt = {}
vehicle_annt_path = "/home/kuriatsu/Documents/PIE/annotations_vehicle/set03/"
for file in glob.glob(os.path.join(vehicle_annt_path, "*obd.xml")):
    filename = file.split("/")[-1].replace(".xml", "")
    video_num = int(filename.split("_")[1])
    tree = ET.parse(file)
    vehicle_annt[video_num] = tree.getroot() 

attrib = {}
attrib_path = "/home/kuriatsu/Documents/PIE/annotations_attributes/set03/"
for file in glob.glob(os.path.join(attrib_path, "*attributes.xml")):
    filename = file.split("/")[-1].replace(".xml", "")
    video_num = int(filename.split("_")[1])
    tree = ET.parse(file)
    attrib[video_num] = tree.getroot() 

with open("/home/kuriatsu/Documents/PIE/log202203/database.pkl", "rb") as f:
    database = pickle.load(f) 


prob_list = []
prediction_list = []
correct_list = []
response_list = []
first_dist_list = []
last_int_dist_list = []
first_speed_list = []
last_int_speed_list = []

log_data["correct"] = None
log_data["response"] = None
log_data["result"] = None
log_data["prob"] = None
log_data["start_dist"] = None
log_data["int_dist"] = None
log_data["start_vehicle_speed"] = None
log_data["int_vehicle_speed"] = None
log_data["future_direction"] = None

for idx, row in log_data.iterrows():
    if row.id in tl_black_list:
        last_state = -2
    else:
        last_state = row.last_state

    if last_state == -1: # no intervention
        correct = -1
        responce = -1

    elif int(row.last_state) == int(row.state):
        if row.id in opposite_anno_list:
            correct = 1
            if last_state == 1:
                responce = 3
            elif last_state == 0:
                responce = 0
            else:
                print(f"last_state{row.last_state}, state{row.state}")
                responce = 3 # ignored=4
        else:
            correct = 0
            if last_state == 1:
                responce = 1
            elif last_state == 0:
                responce = 2
            else:
                print(f"last_state{row.last_state}, state{row.state}")
                responce = 4 # ignored=4
    else:
        if row.id in opposite_anno_list:
            correct = 0
            if last_state == 1:
                responce = 1
            elif last_state == 0:
                responce = 2
            else:
                print(f"last_state{row.last_state}, state{row.state}")
                responce = 4 # ignored=4
        else:
            correct = 1
            if last_state == 1:
                responce = 3
            elif last_state == 0:
                responce = 0
            else:
                print(f"last_state{row.last_state}, state{row.state}")
                responce = 4 # ignored=4
    
    log_data.at[idx, "last_state"] = last_state
    log_data.at[idx, "response"] = responce
    log_data.at[idx, "correct"] = correct

    key = str(row.id)+"_"+str(row.int_length) if row.task == "tl" else str(row.id)+row.task+"_"+str(row.int_length)
    video_num = int(row.id.split("_")[1])
    target_database = database[key]
    
    try:
        if row.task == "tl":
            log_data.at[idx, "prob"] = None
            crossing_frame = target_database["end_frame"]
        else:
            crossing_frame, num_lanes, signalized = get_attrib(attrib[video_num], row.id)
            log_data.at[idx, "prob"] = target_database["prob"] 

        log_data.at[idx, "result"] = target_database["likelihood"]
        log_data.at[idx, "start_dist"] = get_distance(vehicle_annt[video_num], int(target_database["start_frame"]), int(crossing_frame))
        log_data.at[idx, "start_vehicle_speed"] = vehicle_annt[video_num][int(target_database["start_frame"])].attrib.get("OBD_speed")
        log_data.at[idx, "future_direction"] = target_database["future_direction"]

        if int(row.last_state) not in [-1, -2]:
            log_data.at[idx, "int_dist"] = get_distance(vehicle_annt[video_num], int(row.last_int_frame)+int(target_database["start_frame"]), int(crossing_frame))
            log_data.at[idx, "int_vehicle_speed"] = vehicle_annt[video_num][int(row.last_int_frame)+int(target_database["start_frame"])].attrib.get("OBD_speed")

    except Exception as e:
        # print(e, target_database, row, row.last_state, int(row.last_state)==-1)
        print(e, target_database, crossing_frame)


log_data.to_csv("summary.csv")
with open("summary.pkl", "wb") as f:
    pickle.dump(log_data, f)
