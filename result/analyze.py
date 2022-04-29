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
            acc = len(target[target.correct == 1])/(len(target[target.correct == 0]) + len(target[target.correct == 1])+eps)
            missing = len(target[target.correct == -1])/(len(target[target.correct != -2])+eps)
            buf = pd.DataFrame([(subject, task_list.get(task), acc, length, missing)], columns=subject_data.columns)
            subject_data = pd.concat([subject_data, buf])

subject_data.acc = subject_data.acc * 100
subject_data.missing = subject_data.missing * 100
# sns.barplot(x="task", y="acc", hue="int_length", data=subject_data, ci="sd")
# sns.barplot(x="task", y="acc", data=subject_data, ci="sd")

################################################
print("check intervene acc")
################################################
for length in subject_data.int_length.drop_duplicates():
    print(f"acc : length={length}")
    target_df = subject_data[subject_data.int_length == length]
    _, norm_p = stats.shapiro(target_df.acc.dropna())
    _, var_p = stats.levene(
        target_df[target_df.task == 'int'].acc.dropna(),
        target_df[target_df.task == 'traj'].acc.dropna(),
        target_df[target_df.task == 'tl'].acc.dropna(),
        center='median'
        )

    if norm_p < 0.05 or var_p < 0.05:
        print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='acc', group_col='type'))
    else:
        multicomp_result = multicomp.MultiComparison(np.array(target_df.dropna(how='any').acc, dtype="float64"), target_df.dropna(how='any').type)
        print('levene', multicomp_result.tukeyhsd().summary())

fig, ax = plt.subplots()
sns.pointplot(x="int_length", y="acc", data=subject_data, hue="task", hue_order=hue_order, ax=ax, capsize=0.1, ci=95)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel("intervention time [s]", fontsize=18)
ax.set_ylabel("intervention accuracy [%]", fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
plt.show()

################################################
print("check miss rate")
################################################
for length in subject_data.int_length.drop_duplicates():
    print(f"miss : length={length}")
    target_df = subject_data[subject_data.int_length == length]
    _, norm_p = stats.shapiro(target_df.missing.dropna())
    _, var_p = stats.levene(
        target_df[target_df.task == 'int'].missing.dropna(),
        target_df[target_df.task == 'traj'].missing.dropna(),
        target_df[target_df.task == 'tl'].missing.dropna(),
        center='median'
        )

    if norm_p < 0.05 or var_p < 0.05:
        print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='missing', group_col='type'))
    else:
        multicomp_result = multicomp.MultiComparison(np.array(target_df.dropna(how='any').missing, dtype="float64"), target_df.dropna(how='any').type)
        print('levene', multicomp_result.tukeyhsd().summary())

fig, ax = plt.subplots()
sns.pointplot(x="int_length", y="missing", data=subject_data, hue="task", hue_order=hue_order, ax = ax, capsize=0.1, ci=95)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel("intervention time [s]", fontsize=18)
ax.set_ylabel("intervention missing rate [%]", fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)
plt.show()

#####################################
# mean val show
#####################################
target = subject_data[subject_data.task == "crossing intention"]
print("int acc mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].acc.mean(),
    target[target.int_length == 3.0].acc.mean(),
    target[target.int_length == 5.0].acc.mean(),
    target[target.int_length == 8.0].acc.mean(),
    target[target.int_length == 1.0].acc.std(),
    target[target.int_length == 3.0].acc.std(),
    target[target.int_length == 5.0].acc.std(),
    target[target.int_length == 8.0].acc.std(),
))

target = subject_data[subject_data.task == "trajectory"]
print("traj acc mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].acc.mean(),
    target[target.int_length == 3.0].acc.mean(),
    target[target.int_length == 5.0].acc.mean(),
    target[target.int_length == 8.0].acc.mean(),
    target[target.int_length == 1.0].acc.std(),
    target[target.int_length == 3.0].acc.std(),
    target[target.int_length == 5.0].acc.std(),
    target[target.int_length == 8.0].acc.std(),
))

target = subject_data[subject_data.task == "traffic light"]
print("tl acc mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].acc.mean(),
    target[target.int_length == 3.0].acc.mean(),
    target[target.int_length == 5.0].acc.mean(),
    target[target.int_length == 8.0].acc.mean(),
    target[target.int_length == 1.0].acc.std(),
    target[target.int_length == 3.0].acc.std(),
    target[target.int_length == 5.0].acc.std(),
    target[target.int_length == 8.0].acc.std(),
))

target = subject_data[subject_data.task == "crossing intention"]
print("int missing mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].missing.mean(),
    target[target.int_length == 3.0].missing.mean(),
    target[target.int_length == 5.0].missing.mean(),
    target[target.int_length == 8.0].missing.mean(),
    target[target.int_length == 1.0].missing.std(),
    target[target.int_length == 3.0].missing.std(),
    target[target.int_length == 5.0].missing.std(),
    target[target.int_length == 8.0].missing.std(),
))

target = subject_data[subject_data.task == "trajectory"]
print("traj missing mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].missing.mean(),
    target[target.int_length == 3.0].missing.mean(),
    target[target.int_length == 5.0].missing.mean(),
    target[target.int_length == 8.0].missing.mean(),
    target[target.int_length == 1.0].missing.std(),
    target[target.int_length == 3.0].missing.std(),
    target[target.int_length == 5.0].missing.std(),
    target[target.int_length == 8.0].missing.std(),
))

target = subject_data[subject_data.task == "traffic light"]
print("tl missing mean: 1.0:{}, 3.0:{}, 5.0:{}, 8.0:{}\n std {} {} {} {}".format(
    target[target.int_length == 1.0].missing.mean(),
    target[target.int_length == 3.0].missing.mean(),
    target[target.int_length == 5.0].missing.mean(),
    target[target.int_length == 8.0].missing.mean(),
    target[target.int_length == 1.0].missing.std(),
    target[target.int_length == 3.0].missing.std(),
    target[target.int_length == 5.0].missing.std(),
    target[target.int_length == 8.0].missing.std(),
))
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

pd.set_option("max_rows", None)
sort_val = id_data.sort_values(["false_rate","total"], ascending=False)
false_playlist = sort_val[(sort_val.false_rate>0.0)&(sort_val.total>1)]
print(false_playlist)
false_playlist.to_csv("/home/kuriatsu/Dropbox/data/pie202203/false_playlist.csv")

# sns.barplot(x="id", y="acc", hue="int_length", data=id_data)

###############################################
# Workload
###############################################

workload = pd.read_csv("{}/workload.csv".format(data_path))
workload.satisfy = 10-workload.satisfy
workload_melted = pd.melt(workload, id_vars=["subject", "type"], var_name="scale", value_name="score")
#### nasa-tlx ####
for item in workload_melted.scale.drop_duplicates():
    print(item)
    _, norm_p1 = stats.shapiro(workload[workload.type == "int"][item])
    _, norm_p2 = stats.shapiro(workload[workload.type == "traj"][item])
    _, norm_p3 = stats.shapiro(workload[workload.type == "tl"][item])
    _, var_p = stats.levene(
        workload[workload.experiment_type == "int"][item],
        workload[workload.experiment_type == "traj"][item],
        workload[workload.experiment_type == "tl"][item],
        center='median'
        )

    if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
        _, anova_p = stats.friedmanchisquare(
            workload[workload.experiment_type == "int"][item],
            workload[workload.experiment_type == "traj"][item],
            workload[workload.experiment_type == "tl"][item],
        )
        print("anova(friedman test)", anova_p)
        if anova_p < 0.05:
            print(sp.posthoc_conover(workload, val_col=item, group_col="type"))
    else:
        melted_df = pd.melt(nasa_df, id_vars=["name", "experiment_type"],  var_name="type", value_name="rate")
        aov = stats_anova.AnovaRM(workload_melted[workload_melted.type == item], "score", "subject", ["type"])
        print("reperted anova: ", aov.fit())
        multicomp_result = multicomp.MultiComparison(workload_melted[item], nasa_df.type)
        print(multicomp_result.tukeyhsd().summary())

fig, ax = plt.subplots()
sns.barplot(x="scale", y="rate", data=workload_melted, hue="type", hue_order=hue_order, ax=ax)
ax.set_ylim(0, 10)
ax.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left', fontsize=14)
ax.set_xlabel("scale", fontsize=18)
ax.set_ylabel("score (lower is better)", fontsize=18)
ax.tick_params(labelsize=14)
plt.show()
