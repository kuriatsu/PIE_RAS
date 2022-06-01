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
import scikit_posthocs as sp
from scipy import stats
import os
from scipy import stats
import scikit_posthocs as sp


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
    print("{}".format(filename))

    if count in [0, 1, 2]:
        print("skipped")
        continue

    trial = filename.split("_")[-1].replace(".csv", "")
    buf["subject"] = filename.replace("log_data_", "").split("_")[0]
    buf["task"] = filename.replace("log_data_", "").split("_")[1]
    correct_list = []
    response_list = []
    for idx, row in buf.iterrows():
        if row.id in tl_black_list:
            row.last_state = -2
        if row.last_state == -1: # no intervention
            correct_list.append(-1)
            response_list.append(-1)

        elif int(row.last_state) == int(row.state):
            if row.id in opposite_anno_list:
                correct_list.append(1)
                if row.last_state == 1:
                    response_list.append(3)
                elif row.last_state == 0:
                    response_list.append(0)
                else:
                    print(f"last_state{row.last_state}, state{row.state}")
                    response_list.append(4) # ignored=4
            else:
                correct_list.append(0)
                if row.last_state == 1:
                    response_list.append(1)
                elif row.last_state == 0:
                    response_list.append(2)
                else:
                    print(f"last_state{row.last_state}, state{row.state}")
                    response_list.append(4) # ignored=4
        else:
            if row.id in opposite_anno_list:
                correct_list.append(0)
                if row.last_state == 1:
                    response_list.append(1)
                elif row.last_state == 0:
                    response_list.append(2)
                else:
                    print(f"last_state{row.last_state}, state{row.state}")
                    response_list.append(4) # ignored=4
            else:
                correct_list.append(1)
                if row.last_state == 1:
                    response_list.append(3)
                elif row.last_state == 0:
                    response_list.append(0)
                else:
                    print(f"last_state{row.last_state}, state{row.state}")
                    response_list.append(4) # ignored=4

    buf["correct"] = correct_list
    buf["response"] = response_list
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
        target_df[target_df.task == 'trajectory'].acc.dropna(),
        target_df[target_df.task == 'crossing intention'].acc.dropna(),
        target_df[target_df.task == 'traffic light'].acc.dropna(),
        center='median'
        )

    # if norm_p < 0.05 or var_p < 0.05:
    #     print(f"norm:{norm_p}, var:{var_p}")
    #     print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='acc', group_col='task'))
    # else:
    #     multicomp_result = multicomp.MultiComparison(np.array(target_df.dropna(how='any').acc, dtype="float64"), target_df.dropna(how='any').type)
    #     print(f"norm:{norm_p}, var:{var_p}")
    #     print('levene', multicomp_result.tukeyhsd().summary())
    if norm_p < 0.05 or var_p < 0.05:
        _, anova_p = stats.friedmanchisquare(
            target_df[target_df.task == "trajectory"].acc,
            target_df[target_df.task == "crossing intention"].acc,
            target_df[target_df.task == "traffic light"].acc,
        )
        print(f"norm:{norm_p}, var:{var_p}")
        print("anova(friedman test)", anova_p)
        if anova_p < 0.05:
            print('conover\n', sp.posthoc_conover(target_df, val_col="acc", group_col="task"))
            print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='acc', group_col='task'))
    else:
        # melted_df = pd.melt(target_df, id_vars=["subject", "acc", "int_length"],  var_name="task", value_name="rate")
        aov = stats_anova.AnovaRM(melted_df, "missing", "subject", ["task"])
        print(f"norm:{norm_p}, var:{var_p}")
        print("reperted anova: ", aov.fit())
        multicomp_result = multicomp.MultiComparison(melted_df[length], nasa_df.task)
        print(melted_df.tukeyhsd().summary())

fig, ax = plt.subplots()
sns.pointplot(x="int_length", y="acc", data=subject_data, hue="task", hue_order=hue_order, ax=ax, capsize=0.1, ci="sd")
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
        target_df[target_df.task == 'trajectory'].missing.dropna(),
        target_df[target_df.task == 'crossing intention'].missing.dropna(),
        target_df[target_df.task == 'traffic light'].missing.dropna(),
        center='median'
        )

    # if norm_p < 0.05 or var_p < 0.05:
    #     print(f"norm:{norm_p}, var:{var_p}")
    #     print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='missing', group_col='task'))
    # else:
    #     multicomp_result = multicomp.MultiComparison(np.array(target_df.dropna(how='any').missing, dtype="float64"), target_df.dropna(how='any').type)
    #     print(f"norm:{norm_p}, var:{var_p}")
    #     print('levene', multicomp_result.tukeyhsd().summary())

    if norm_p < 0.05 or var_p < 0.05:
        _, anova_p = stats.friedmanchisquare(
            target_df[target_df.task == "trajectory"].missing,
            target_df[target_df.task == "crossing intention"].missing,
            target_df[target_df.task == "traffic light"].missing,
        )
        print(f"norm:{norm_p}, var:{var_p}")
        print("anova(friedman test)", anova_p)
        if anova_p < 0.05:
            print('steel-dwass\n', sp.posthoc_dscf(target_df, val_col='missing', group_col='task'))
            print('conover\n', sp.posthoc_conover(target_df, val_col="missing", group_col="task"))
    else:
        # melted_df = pd.melt(target_df, id_vars=["subject", "acc", "int_length"],  var_name="task", value_name="rate")
        aov = stats_anova.AnovaRM(melted_df, "missing", "subject", ["task"])
        print(f"norm:{norm_p}, var:{var_p}")
        print("reperted anova: ", aov.fit())
        multicomp_result = multicomp.MultiComparison(melted_df[length], nasa_df.task)
        print(melted_df.tukeyhsd().summary())

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

###############################################################################################
print("response rate stacked bar plot")
###############################################################################################
response_summary_pred = pd.DataFrame(columns=["int_length", "task", "response", "count"])
for int_length in log_data.int_length.drop_duplicates():
    for task in log_data.task.drop_duplicates():
        for response in [0, 1, 2, 3, -1]:
            buf = pd.Series([int_length, task, response,
                len(log_data[(log_data.int_length==int_length) & (log_data.task==task) & (log_data.response <= response)])/len(log_data[(log_data.int_length==int_length) & (log_data.task==task) & (log_data.response!=4)])],
                index=response_summary_pred.columns)
            response_summary_pred = response_summary_pred.append(buf, ignore_index=True)

fig, axes = plt.subplots()

cr = sns.barplot(x="task", y="count", hue="int_length", data=response_summary_pred[response_summary_pred.response==3], ax=axes, palette=sns.color_palette(["turquoise"]*6), edgecolor="0.2", order=["tl", "int", "traj"])
fa = sns.barplot(x="task", y="count", hue="int_length", data=response_summary_pred[response_summary_pred.response==2], ax=axes, palette=sns.color_palette(["orangered"]*6), edgecolor="0.2", order=["tl", "int", "traj"])
miss = sns.barplot(x="task", y="count", hue="int_length", data=response_summary_pred[response_summary_pred.response==1], ax=axes, palette=sns.color_palette(["lightsalmon"]*6), edgecolor="0.2", order=["tl", "int", "traj"])
hit = sns.barplot(x="task", y="count", hue="int_length", data=response_summary_pred[response_summary_pred.response==0], ax=axes, palette=sns.color_palette(["teal"]*6), edgecolor="0.2", order=["tl", "int", "traj"])
no_int = sns.barplot(x="task", y="count", hue="int_length", data=response_summary_pred[response_summary_pred.response==-1], ax=axes, palette=sns.color_palette(["gray"]*6), edgecolor="0.2", order=["tl", "int", "traj"])
axes.set_xticks([-0.3, -0.1, 0.1, 0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3])
axes.set_xticklabels(["1.0", "3.0", "5.0", "8.0", "1.0", "3.0", "5.0", "8.0", "1.0", "3.0", "5.0", "8.0"], fontsize=14)
# axes.set_yticklabels(fontsize=14)
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.75, ax_pos.y1-0.84, "traffic light", fontsize=14)
fig.text(ax_pos.x1-0.55, ax_pos.y1-0.84, "crossing intention", fontsize=14)
fig.text(ax_pos.x1-0.25, ax_pos.y1-0.84, "trajectory", fontsize=14)
axes.tick_params(labelsize=14)
axes.set_ylabel("Response Rate", fontsize=18)
axes.set_xlabel("")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[::4], ["CR", "FA", "miss", "hit", "no_int"], bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=14)
plt.show()

###############################################
# Workload
###############################################

workload = pd.read_csv("{}/workload.csv".format(data_path))
workload.satisfy = 10-workload.satisfy
workload_melted = pd.melt(workload, id_vars=["subject", "type"], var_name="scale", value_name="score")
#### nasa-tlx ####
for item in workload_melted.scale.drop_duplicates():
    print(item)
    _, norm_p1 = stats.shapiro(workload[workload.type == "trajectory"][item])
    _, norm_p2 = stats.shapiro(workload[workload.type == "crossing intention"][item])
    _, norm_p3 = stats.shapiro(workload[workload.type == "traffic light"][item])
    _, var_p = stats.levene(
        workload[workload.type == "trajectory"][item],
        workload[workload.type == "crossing intention"][item],
        workload[workload.type == "traffic light"][item],
        center='median'
        )

    if norm_p1 < 0.05 or norm_p2 < 0.05 or norm_p3 < 0.05 or norm_p4 < 0.05:
        _, anova_p = stats.friedmanchisquare(
            workload[workload.type == "trajectory"][item],
            workload[workload.type == "crossing intention"][item],
            workload[workload.type == "traffic light"][item],
        )
        print("anova(friedman test)", anova_p)
        if anova_p < 0.05:
            print(sp.posthoc_conover(workload, val_col=item, group_col="type"))
    else:
        melted_df = pd.melt(nasa_df, id_vars=["name", "experiment_type"],  var_name="type", value_name="score")
        aov = stats_anova.AnovaRM(workload_melted[workload_melted.type == item], "score", "subject", ["type"])
        print("reperted anova: ", aov.fit())
        multicomp_result = multicomp.MultiComparison(workload_melted[item], nasa_df.type)
        print(multicomp_result.tukeyhsd().summary())

fig, ax = plt.subplots()
sns.barplot(x="scale", y="score", data=workload_melted, hue="type", hue_order=hue_order, ax=ax)
ax.set_ylim(0, 10)
ax.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left', fontsize=14)
ax.set_xlabel("scale", fontsize=18)
ax.set_ylabel("score (lower is better)", fontsize=18)
ax.tick_params(labelsize=14)
plt.show()

###############################################
# necessary time
###############################################
time = pd.read_csv("/home/kuriatsu/Dropbox/documents/subjective_time.csv")
fig, ax = plt.subplots()
# mean_list = [
#     time[time.type=="crossing intention"].ideal_time.mean(),
#     time[time.type=="trajectory"].ideal_time.mean(),
#     time[time.type=="traffic light"].ideal_time.mean(),
# ]
# sem_list = [
#     time[time.type=="crossing intention"].ideal_time.sem(),
#     time[time.type=="trajectory"].ideal_time.sem(),
#     time[time.type=="traffic light"].ideal_time.sem(),
# ]
_, norm_p = stats.shapiro(time.ideal_time.dropna())
_, var_p = stats.levene(
    time[time.type == 'crossing intention'].ideal_time.dropna(),
    time[time.type == 'trajectory'].ideal_time.dropna(),
    time[time.type == 'traffic light'].ideal_time.dropna(),
    center='median'
    )

if norm_p < 0.05 or var_p < 0.05:
    print('steel-dwass\n', sp.posthoc_dscf(time, val_col='ideal_time', group_col='type'))
else:
    multicomp_result = multicomp.MultiComparison(np.array(time.dropna(how='any').ideal_time, dtype="float64"), time.dropna(how='any').type)
    print('levene', multicomp_result.tukeyhsd().summary())

sns.pointplot(x="type", y="ideal_time", hue="type", hue_order=hue_order, data=time, join=False, ax=ax, capsize=0.1, ci=95)
ax.set_ylim(0.5,3.5)
plt.yticks([1, 2, 3, 4], ["<3", "3-5", "5-8", "8<"])
plt.show()

###############################################
# compare prediction and intervention
###############################################
with open("/home/kuriatsu/Dropbox/data/pie202203/database.pkl", "rb") as f:
    database = pickle.load(f)

tl_result = pd.read_csv("/home/kuriatsu/Dropbox/data/pie202203/tlr_result.csv")

overall_result = pd.DataFrame(columns=["id", "task", "subject", "gt", "int", "prediction"])

log_data = None
data_path = "/home/kuriatsu/Dropbox/data/pie202203"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    buf = pd.read_csv(file)
    filename =file.split("/")[-1]
    count = float(filename.replace("log_data_", "").split("_")[-1].replace(".csv", ""))
    print("{}".format(filename))

    if count in [0, 1, 2]:
        print("skipped")
        continue

    subject = filename.replace("log_data_", "").split("_")[0]
    task = filename.replace("log_data_", "").split("_")[1]
    for idx, row in buf.iterrows():

        if task != "tl":
            database_id = row.id+task+"_"+str(float(row.int_length))
            prediction = (database[database_id].get("likelihood") <= 0.5)
            gt = False if row.state else True
        else:
            database_id = row.id+"_"+str(float(row.int_length))
            prediction = 1 if float(tl_result[tl_result.id == row.id].result) == 2 else 0
            gt = False if row.state else True

        if row.id in tl_black_list:
            intervention = -2
        if row.last_state == -1: # no intervention
            intervention = -1

        else:
            if row.id in opposite_anno_list:
                intervention = False if row.last_state else True
            else:
                intervention = row.last_state

        buf = pd.DataFrame([(row.id, task, subject, int(gt), int(intervention), int(prediction))], columns = overall_result.columns)
        overall_result = pd.concat([overall_result, buf])

overall_result.to_csv("/home/kuriatsu/Dropbox/data/pie202203/acc.csv")
