#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()

def getAtrrib(root, id):
    for pedestrian in root.iter("pedestrian"):
        if pedestrian.attrib.get("id") == id:
            return pedestrian

    return None


base_dir = "/media/kuriatsu/SamsungKURI1/PIE_data"
# base_dir = "/home/kuriatsu/Documents/experiment/pie"

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

play_list = pd.read_csv("{}/extracted_data/playlist.csv".format(base_dir))
summary = pd.DataFrame(columns=["id", "thres", "last_state", "int_result", "int_time", "prediction_value", "prediction_bin", "annotation_value", "annotation_bin", "crossing", "limit"])
summary_list = []
# data = pd.read_csv("{}/202201_pre_experiment/log_data_1_kanayama_nocon.csv".format(base_dir))
data = pd.read_csv("{}/extracted_data/log_data_1_kanayama_nocon.csv".format(base_dir))
for i, row in data.iterrows():
    for video_name in play_list.iloc[0, :]:
        if video_name.startswith(row.id):
            int_time = video_name.split("_")[-1]
    summary_list.append([row.id, 0.5, row.last_state, 0, float(row.last_int_frame)/30.0, 0, 0, 0, 0, 0, float(int_time)])

# data = pd.read_csv("{}/202201_pre_experiment/log_data_2_kanayama_con.csv".format(base_dir))
data = pd.read_csv("{}/extracted_data/log_data_2_kanayama_con.csv".format(base_dir))
for i, row in data.iterrows():
    for video_name in play_list.iloc[1, :]:
        if video_name.startswith(row.id):
            int_time = video_name.split("_")[-1]
    summary_list.append([row.id, 0.0, row.last_state, 0, float(row.last_int_frame)/30.0, 0, 0, 0, 0, 0, float(int_time)])

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
    base_dir + "/extracted_data/predict/test/result_0-150.pkl",
    base_dir + "/extracted_data/predict/test/result_150-300.pkl",
    base_dir + "/extracted_data/predict/test/result_300-450.pkl",
    base_dir + "/extracted_data/predict/test/result_450-600.pkl",
    base_dir + "/extracted_data/predict/test/result_600-719.pkl",
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
# print(summary_list)
total_acc = [
[row[3] for row in summary_list if row[10]==0.0].count(1)/len([row[3] for row in summary_list if row[10]==0.0]),
[row[3] for row in summary_list if row[10]==1.0].count(1)/len([row[3] for row in summary_list if row[10]==1.0]),
[row[3] for row in summary_list if row[10]==3.0].count(1)/len([row[3] for row in summary_list if row[10]==3.0]),
[row[3] for row in summary_list if row[10]==5.0].count(1)/len([row[3] for row in summary_list if row[10]==5.0]),
[row[3] for row in summary_list if row[10]==7.0].count(1)/len([row[3] for row in summary_list if row[10]==7.0]),
[row[3] for row in summary_list if row[10]==9.0].count(1)/len([row[3] for row in summary_list if row[10]==9.0]),
]
print("int limit time and acc", total_acc)
time_acc = [
len([row[3] for row in summary_list if (row[10]==0.0 and not math.isnan(row[4]))]),
len([row[3] for row in summary_list if (row[10]==1.0 and not math.isnan(row[4]))]),
len([row[3] for row in summary_list if (row[10]==3.0 and not math.isnan(row[4]))]),
len([row[3] for row in summary_list if (row[10]==5.0 and not math.isnan(row[4]))]),
len([row[3] for row in summary_list if (row[10]==7.0 and not math.isnan(row[4]))]),
len([row[3] for row in summary_list if (row[10]==9.0 and not math.isnan(row[4]))]),
]
print("intervention count",time_acc)
int_acc = [
[row[3] for row in summary_list if (row[10]==0.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==0.0 and not math.isnan(row[4]))]),
[row[3] for row in summary_list if (row[10]==1.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==1.0 and not math.isnan(row[4]))]),
[row[3] for row in summary_list if (row[10]==3.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==3.0 and not math.isnan(row[4]))]),
[row[3] for row in summary_list if (row[10]==5.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==5.0 and not math.isnan(row[4]))]),
[row[3] for row in summary_list if (row[10]==7.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==7.0 and not math.isnan(row[4]))]),
[row[3] for row in summary_list if (row[10]==9.0 and not math.isnan(row[4]))].count(1)/len([row[3] for row in summary_list if (row[10]==9.0 and not math.isnan(row[4]))]),
]
print("intervention acc", int_acc)
cross_rate = [
[row[8] for row in summary_list if row[10]==0.0].count(1)/len([row[8] for row in summary_list if row[10]==0.0]),
[row[8] for row in summary_list if row[10]==1.0].count(1)/len([row[8] for row in summary_list if row[10]==1.0]),
[row[8] for row in summary_list if row[10]==3.0].count(1)/len([row[8] for row in summary_list if row[10]==3.0]),
[row[8] for row in summary_list if row[10]==5.0].count(1)/len([row[8] for row in summary_list if row[10]==5.0]),
[row[8] for row in summary_list if row[10]==7.0].count(1)/len([row[8] for row in summary_list if row[10]==7.0]),
[row[8] for row in summary_list if row[10]==9.0].count(1)/len([row[8] for row in summary_list if row[10]==9.0]),
]
print("annot cross", cross_rate)

pred_acc = [
[row[6] for row in summary_list if row[10]==0.0].count(1)/len([row[6] for row in summary_list if row[10]==0.0]),
[row[6] for row in summary_list if row[10]==1.0].count(1)/len([row[6] for row in summary_list if row[10]==1.0]),
[row[6] for row in summary_list if row[10]==3.0].count(1)/len([row[6] for row in summary_list if row[10]==3.0]),
[row[6] for row in summary_list if row[10]==5.0].count(1)/len([row[6] for row in summary_list if row[10]==5.0]),
[row[6] for row in summary_list if row[10]==7.0].count(1)/len([row[6] for row in summary_list if row[10]==7.0]),
[row[6] for row in summary_list if row[10]==9.0].count(1)/len([row[6] for row in summary_list if row[10]==9.0]),
]
print("prediction cross", pred_acc)

time_acc = [
[row[6] == row[8] for row in summary_list if row[10]==0.0].count(1)/len([row[6] for row in summary_list if row[10]==0.0]),
[row[6] == row[8] for row in summary_list if row[10]==1.0].count(1)/len([row[6] for row in summary_list if row[10]==1.0]),
[row[6] == row[8] for row in summary_list if row[10]==3.0].count(1)/len([row[6] for row in summary_list if row[10]==3.0]),
[row[6] == row[8] for row in summary_list if row[10]==5.0].count(1)/len([row[6] for row in summary_list if row[10]==5.0]),
[row[6] == row[8] for row in summary_list if row[10]==7.0].count(1)/len([row[6] for row in summary_list if row[10]==7.0]),
[row[6] == row[8] for row in summary_list if row[10]==9.0].count(1)/len([row[6] for row in summary_list if row[10]==9.0]),
]
print("prediction acc", time_acc)

time_acc = [
sum([row[4] for row in summary_list if (row[10]==0.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==0.0 and not math.isnan(row[4]))]),
sum([row[4] for row in summary_list if (row[10]==1.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==1.0 and not math.isnan(row[4]))]),
sum([row[4] for row in summary_list if (row[10]==3.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==3.0 and not math.isnan(row[4]))]),
sum([row[4] for row in summary_list if (row[10]==5.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==5.0 and not math.isnan(row[4]))]),
sum([row[4] for row in summary_list if (row[10]==7.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==7.0 and not math.isnan(row[4]))]),
sum([row[4] for row in summary_list if (row[10]==9.0 and not math.isnan(row[4]))])/len([row[4] for row in summary_list if (row[10]==9.0 and not math.isnan(row[4]))]),
]
print("intervention time", time_acc)

left = np.arange(len(int_acc))
fig, ax = plt.subplots()
ax.bar(x=left, height=int_acc, width=0.4, align="center", label="intervention")
ax.bar(x=left+0.4, height=pred_acc, width=0.4, align="center", label="prediction")
ax.bar(x=left+0.8, height=total_acc, width=0.4, align="center", label="total")
ax.set_xticks(left+0.2)
ax.set_xticklabels(labels=[0.0, 1.0, 3.0, 5.0, 7.0, 9.0])
plt.show()
