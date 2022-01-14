#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math

def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()

def getAtrrib(root, id):
    for pedestrian in root.iter("pedestrian"):
        if pedestrian.attrib.get("id") == id:
            return pedestrian

    return None


base_dir = "/home/kuriatsu/Documents/experiment/pie"
video_list = [
    "set03/video_0001",
    "set03/video_0002",
    "set03/video_0003",
    "set03/video_0004",
    "set03/video_0005",
    "set03/video_0006",
    "set03/video_0007",
    "set03/video_0008",
    "set03/video_0009",
    "set03/video_0010",
    "set03/video_0011",
    "set03/video_0012",
    "set03/video_0013",
    "set03/video_0014",
    "set03/video_0015",
    "set03/video_0016",
    "set03/video_0017",
    "set03/video_0018",
    "set03/video_0019",
    # "set05/video_0001",
    # "set05/video_0002",
]

summary = pd.DataFrame(columns=["id", "thres", "last_state", "int_result", "int_time", "prediction_value", "prediction_bin", "annotation_value", "annotation_bin", "crossing"])
summary_list = []
data = pd.read_csv("{}/202201_pre_experiment/log_data_1_kanayama_nocon.csv".format(base_dir))
for i, row in data.iterrows():
    summary_list.append([row.id, 0.5, row.last_state, 0, row.last_int_time, 0, 0, 0, 0, 0])

data = pd.read_csv("{}/202201_pre_experiment/log_data_2_kanayama_con.csv".format(base_dir))
for i, row in data.iterrows():
    summary_list.append([row.id, 0.0, row.last_state, 0, row.last_int_time, 0, 0, 0, 0, 0])

for video_name in video_list:
    root = getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_name))
    for pedestrian in root.iter("pedestrian"):
        for row in summary_list:
            if row[0] == pedestrian.get("id"):
                row[7] = float(pedestrian.get("intention_prob"))
                row[8] = 0 if float(pedestrian.get("intention_prob")) < 0.5 else 1
                row[9] = int(float(pedestrian.get("crossing")))
                row[9] = 0 if row[9] == -1 else row[9]

result_file_list = [
    base_dir + "/pie_predict/test/result_0-150.pkl",
    base_dir + "/pie_predict/test/result_151-300.pkl",
    base_dir + "/pie_predict/test/result_301-450.pkl",
    base_dir + "/pie_predict/test/result_451-600.pkl",
    base_dir + "/pie_predict/test/result_601-719.pkl",
    # base_dir + "/extracted_data/predict/val/result_0-150.pkl",
    # base_dir + "/extracted_data/predict/val/result_151-243.pkl",
    ]

prediction_data = []
for file in result_file_list:
    with open(file, "rb") as f:
        prediction_data+=pickle.load(f)

# get final prediction result for each pedestrian
for buf in prediction_data:
    for row in summary_list:
        if row[0] == buf.get("ped_id"):
            row[5] = float(buf.get("res"))
            row[6] = 0 if float(buf.get("res")) < 0.5 else 1

for row in summary_list:
    if (row[5] < row[1] and row[2] == False) or (row[5] >= row[1] and row[2] == True):
        row[3] = 0
    else:
        row[3] = 1

[r[3] for r in summary_list].count(0)

acc_pred_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
acc_pred_nor_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
acc_pred_con_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
for row in summary_list:
    acc_pred_mat.at[row[6], row[8]] += 1
    if row[1] == 0.5:
        acc_pred_nor_mat.at[row[6], row[8]] += 1
    elif row[1] == 0.0:
        acc_pred_con_mat.at[row[6], row[8]] += 1

crossing_pred_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
for row in summary_list:
    crossing_pred_mat.at[row[6], row[9]] += 1

crossing_gt_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
for row in summary_list:
    crossing_gt_mat.at[row[8], row[9]] += 1

acc_int_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
acc_int_con_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
acc_int_nor_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])

for row in summary_list:
    acc_int_mat.at[row[3], row[8]] += 1
    if row[1] == 0.5:
        acc_int_nor_mat.at[row[3], row[8]] += 1
    elif row[1] == 0.0:
        acc_int_con_mat.at[row[3], row[8]] += 1


crossing_int_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
crossing_int_con_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])
crossing_int_nor_mat = pd.DataFrame([[0,0],[0,0]], columns=[1, 0], index=[1, 0])

for row in summary_list:
    crossing_int_mat.at[row[3], row[9]] += 1
    if row[1] == 0.5:
        crossing_int_nor_mat.at[row[3], row[9]] += 1
    elif row[1] == 0.0:
        crossing_int_con_mat.at[row[3], row[9]] += 1
    # if row[5] != row[7]:
    #     row[9] = 0
    #     row[8] = 0 if row[2] == False else 1
    # else:
    #     row[9] = 1
    #     row[8] = 1 if row[2] == False else 0
