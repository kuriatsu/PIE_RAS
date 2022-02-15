#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(context='paper', style='whitegrid')

database = pd.read_csv("/home/kuriatsu/Documents/experiment/pie_202201/summary.csv")
# database = pd.read_csv("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/log/summary.csv")

###############################################################################################
print("responce vs prob count plot x=[0, 1, 2, 3] y=count")
###############################################################################################
fig, axes = plt.subplots(1, 3)
sns.countplot(x="responce_int_vs_prob", hue="length", data=database[database.recognition_thresh==0.0], ax=axes[0])
sns.countplot(x="responce_int_vs_prob", hue="length", data=database[database.recognition_thresh==0.5], ax=axes[1])
sns.countplot(x="responce_int_vs_prob", hue="length", data=database[database.recognition_thresh==0.8], ax=axes[2])

# acc rate calcuration
acc_summary = pd.DataFrame(columns=["length", "thresh", "subject", "acc_int_prob", "acc_pred_prob", "acc_int_cross", "acc_pred_cross"])
for subject in database.subject.drop_duplicates():
    for length in database.length.drop_duplicates():
        for thresh in database.recognition_thresh.drop_duplicates():
            target_db = database[(database.subject == subject) & (database.length == length) & (database.recognition_thresh == thresh)]
            if len(target_db) == 0:
                continue
            buf = pd.Series([
                length,
                thresh,
                subject,
                sum(target_db.acc_int)/len(target_db.acc_int),
                sum(target_db.acc_pred)/len(target_db.acc_pred),
                sum(target_db.acc_int_cross)/len(target_db.acc_int_cross),
                sum(target_db.acc_pred_cross)/len(target_db.acc_pred_cross),
                ], index=acc_summary.columns)
            acc_summary = acc_summary.append(buf, ignore_index=True)

###############################################################################################
print("intervention/prediction accuracy on intention_prob  x=thresh*length y=acc_pred and acc_int")
###############################################################################################
fig, axes = plt.subplots()
sns.barplot(x="thresh", y="acc_pred_prob", hue="length", data=acc_summary, ax=axes, palette="Blues", alpha=1.0)
sns.barplot(x="thresh", y="acc_int_prob", hue="length", data=acc_summary, ax=axes, palette="OrRd", alpha=0.7)
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=15)
axes.set_ylabel("Accuracy")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[5::6], ["prediction", "intervention"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

###############################################################################################
print("intervention/prediction accuracy on crossing  x=thresh*length y=acc_pred and acc_int")
###############################################################################################
fig, axes = plt.subplots()
sns.barplot(x="thresh", y="acc_pred_cross", hue="length", data=acc_summary, ax=axes, palette="Blues", alpha=1.0)
sns.barplot(x="thresh", y="acc_int_cross", hue="length", data=acc_summary, ax=axes, palette="OrRd", alpha=0.7)
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=15)
axes.set_ylabel("Accuracy")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[5::6], ["prediction", "intervention"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()


# *only intervention done* acc rate calcuration
acc_summary_int_done = pd.DataFrame(columns=["length", "thresh", "subject", "acc_int_prob", "acc_int_cross", "int_rate"])
for subject in database.subject.drop_duplicates():
    for length in database.length.drop_duplicates():
        for thresh in database.recognition_thresh.drop_duplicates():
            target_db = database[(database.subject == subject) & (database.length == length) & (database.recognition_thresh == thresh) & (database.int_count > 0)]
            if len(target_db) == 0:
                continue
            buf = pd.Series([
                length,
                thresh,
                subject,
                sum(target_db.acc_int)/len(target_db.acc_int),
                sum(target_db.acc_int_cross)/len(target_db.acc_int_cross),
                len(target_db)/len(database[(database.subject == subject) & (database.length == length) & (database.recognition_thresh == thresh)]),
                ], index=acc_summary_int_done.columns)
            acc_summary_int_done = acc_summary_int_done.append(buf, ignore_index=True)


###############################################################################################
print("*only intervention done* intervention/prediction accuracy on intention_prob  x=thresh*length y=acc_pred and acc_int")
###############################################################################################
fig, axes = plt.subplots()
sns.barplot(x="thresh", y="int_rate", hue="length", data=acc_summary_int_done, ax=axes, palette="Blues", alpha=1.0)
sns.barplot(x="thresh", y="acc_int_prob", hue="length", data=acc_summary_int_done, ax=axes, palette="OrRd", alpha=0.6)
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=15)
axes.set_ylabel("Count")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[5::6], ["intervention rate", "accuracy"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

###############################################################################################
print("*only intervention done* intervention/prediction accuracy on crossing  x=thresh*length y=acc_pred and acc_int")
###############################################################################################
fig, axes = plt.subplots()
sns.barplot(x="thresh", y="int_rate", hue="length", data=acc_summary_int_done, ax=axes, palette="Blues", alpha=1.0)
sns.barplot(x="thresh", y="acc_int_cross", hue="length", data=acc_summary_int_done, ax=axes, palette="OrRd", alpha=0.7)
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=15)
axes.set_ylabel("Accuracy")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[5::6], ["intervention rate", "accuracy"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()



###############################################################################################
print("int vs prob responce rate stacked bar plot")
###############################################################################################
responce_summary_int = pd.DataFrame(columns=["length", "thresh", "responce", "count"])
for length in database.length.drop_duplicates():
    for thresh in database.recognition_thresh.drop_duplicates():
        for responce in [0, 1, 2, 3]:
            buf = pd.Series([length, thresh, responce,
                len(database[(database.length==length) & (database.recognition_thresh==thresh) & (database.responce_int_vs_prob <= responce)])/len(database[(database.length==length) & (database.recognition_thresh==thresh)])],
                index=responce_summary_int.columns)
            responce_summary_int = responce_summary_int.append(buf, ignore_index=True)

fig, axes = plt.subplots()
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_int[responce_summary_int.responce==3], ax=axes, palette=sns.color_palette(["orangered"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_int[responce_summary_int.responce==2], ax=axes, palette=sns.color_palette(["lightsalmon"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_int[responce_summary_int.responce==1], ax=axes, palette=sns.color_palette(["turquoise"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_int[responce_summary_int.responce==0], ax=axes, palette=sns.color_palette(["teal"]*6), edgecolor="0.2")
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=20)
axes.set_ylabel("Responce Rate")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[::6], ["CR", "FA", "miss", "hit"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()


###############################################################################################
print(" prediction responce rate stacked bar plot")
###############################################################################################
responce_summary_pred = pd.DataFrame(columns=["length", "thresh", "responce", "count"])
for length in database.length.drop_duplicates():
    for thresh in database.recognition_thresh.drop_duplicates():
        for responce in [0, 1, 2, 3]:
            buf = pd.Series([length, thresh, responce,
                len(database[(database.length==length) & (database.recognition_thresh==thresh) & (database.responce_pred <= responce)])/len(database[(database.length==length) & (database.recognition_thresh==thresh)])],
                index=responce_summary_pred.columns)
            responce_summary_pred = responce_summary_pred.append(buf, ignore_index=True)

fig, axes = plt.subplots()
cr = sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_pred[responce_summary_pred.responce==3], ax=axes, palette=sns.color_palette(["orangered"]*6), edgecolor="0.2")
fa = sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_pred[responce_summary_pred.responce==2], ax=axes, palette=sns.color_palette(["lightsalmon"]*6), edgecolor="0.2")
miss = sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_pred[responce_summary_pred.responce==1], ax=axes, palette=sns.color_palette(["turquoise"]*6), edgecolor="0.2")
hit = sns.barplot(x="thresh", y="count", hue="length", data=responce_summary_pred[responce_summary_pred.responce==0], ax=axes, palette=sns.color_palette(["teal"]*6), edgecolor="0.2")
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=20)
axes.set_ylabel("Responce Rate")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[::6], ["CR", "FA", "miss", "hit"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

###############################################################################################
print("int vs prediction responce rate stacked bar plot")
###############################################################################################
responce_summary = pd.DataFrame(columns=["length", "thresh", "responce", "count"])
for length in database.length.drop_duplicates():
    for thresh in database.recognition_thresh.drop_duplicates():
        for responce in [0, 1, 2, 3]:
            buf = pd.Series([length, thresh, responce,
                len(database[(database.length==length) & (database.recognition_thresh==thresh) & (database.response_int_vs_pred <= responce)])/len(database[(database.length==length) & (database.recognition_thresh==thresh)])],
                index=responce_summary.columns)
            responce_summary = responce_summary.append(buf, ignore_index=True)

fig, axes = plt.subplots()
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary[responce_summary.responce==3], ax=axes, palette=sns.color_palette(["orangered"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary[responce_summary.responce==2], ax=axes, palette=sns.color_palette(["lightsalmon"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary[responce_summary.responce==1], ax=axes, palette=sns.color_palette(["turquoise"]*6), edgecolor="0.2")
sns.barplot(x="thresh", y="count", hue="length", data=responce_summary[responce_summary.responce==0], ax=axes, palette=sns.color_palette(["teal"]*6), edgecolor="0.2")
axes.set_xticks([-0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.66, 0.8, 0.93, 1.07, 1.2, 1.34, 1.66, 1.8, 1.93, 2.07, 2.2, 2.34], y=-0.5)
axes.set_xticklabels(["1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0", "1.0", "3.0", "5.0", "7.0", "9.0", "12.0"])
ax_pos = axes.get_position()
fig.text(ax_pos.x1-0.7, ax_pos.y1-0.84, "thresh=0.0")
fig.text(ax_pos.x1-0.44, ax_pos.y1-0.84, "thresh=0.5")
fig.text(ax_pos.x1-0.17, ax_pos.y1-0.84, "thresh=0.8")
axes.set_xlabel("Recognition Thres and Intervention Time[s]", labelpad=20)
axes.set_ylabel("Responce Rate")
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles[::6], ["CR", "FA", "miss", "hit"], bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()
