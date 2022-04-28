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
sns.set(context='paper', style='whitegrid')

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
data_path = "/home/kuriatsu/Dropbox/data/pie202203"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    buf = pd.read_csv(file)
    filename =file.split("/")[-1]
    count = int(filename.replace("log_data_", "").split("_")[-1].replace(".csv", ""))
    if count in [0, 1, 2]:
        print("{} skipped".format(filename))
        continue
    trial = filename.split("_")[-1].replace(".csv", "")
    buf["subject"] = filename.replace("log_data_", "").split("_")[0]
    buf["task"] = filename.replace("log_data_", "").split("_")[1]
    correct_list = []
    for idx, row in buf.iterrows():
        if row.id in tl_black_list:
            row.last_state = -2
        if row.last_state == -1:
            correct_list.append(-1)
        elif int(row.last_state) == int(row.state):
            if row.id in opposite_anno_list:
                correct_list.append(1)
            else:
                correct_list.append(0)
        else:
            if row.id in opposite_anno_list:
                correct_list.append(0)
            else:
                correct_list.append(1)

    buf["correct"] = correct_list
    len(correct_list)
    if log_data is None:
        log_data = buf
    else:
        log_data = log_data.append(buf, ignore_index=True)

task_list = {"int": "crossing intention", "tl": "traffic light", "traj":"trajectory"}
subject_data = pd.DataFrame(columns=["subject", "task", "acc", "int_length", "missing"])
for subject in log_data.subject.drop_duplicates():
    for task in log_data.task.drop_duplicates():
        for length in log_data.int_length.drop_duplicates():
            target = log_data[(log_data.subject == subject) & (log_data.task == task) & (log_data.int_length == length)]
            # acc = len(target[target.correct == 1])/(len(target))
            acc = len(target[target.correct == 1])/(len(target[target.correct == 0]) + len(target[target.correct == 1]))
            missing = len(target[target.correct == -1])/len(target[target.correct != -2])
            buf = pd.DataFrame([(subject, task_list.get(task), acc, length, missing)], columns=subject_data.columns)
            subject_data = pd.concat([subject_data, buf])

# sns.barplot(x="task", y="acc", hue="int_length", data=subject_data, ci="sd")
# sns.barplot(x="task", y="acc", data=subject_data, ci="sd")
fig, ax = plt.subplots()
sns.pointplot(x="int_length", y="missing", data=subject_data, hue="task", ax = ax)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("intervention time [s]", fontsize=14)
ax.set_ylabel("intervention missing rate", fontsize=14)
plt.show()
fig, ax = plt.subplots()

sns.pointplot(x="int_length", y="acc", data=subject_data, hue="task", ax=ax)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("intervention time [s]", fontsize=14)
ax.set_ylabel("intervention accuracy", fontsize=14)
plt.show()


###########################################
# collect wrong intervention ids
###########################################

task_list = {"int": "crossing intention", "tl": "traffic light", "traj":"trajectory"}
id_data = pd.DataFrame(columns=["id", "task", "false_rate", "missing", "total"])
for id in log_data.id.drop_duplicates():
    for task in log_data.task.drop_duplicates():
        for length in log_data.int_length.drop_duplicates():
            target = log_data[(log_data.id == id) & (log_data.task == task) & (log_data.int_length == length)]
            # acc = len(target[target.correct == 1])/(len(target))
            total = len(target)
            name = id.replace("tl","")+task+"_"+str(length)
            if len(target) > 0:
                false_rate = len(target[target.correct == 0])/len(target)
            else:
                false_rate = 0.0

            missing = len(target[target.correct == -1])
            buf = pd.DataFrame([(name, task, false_rate, missing, total)], columns=id_data.columns)
            id_data = pd.concat([id_data, buf])

pd.set_option("max_rows", 100)
sort_val = id_data.sort_values(["false_rate","total"], ascending=False)
false_playlist = sort_val[(sort_val.false_rate>0.0)&(sort_val.total>1)]
false_playlist.to_csv("/home/kuriatsu/Dropbox/data/pie202203/false_playlist.csv")

# sns.barplot(x="id", y="acc", hue="int_length", data=id_data)
