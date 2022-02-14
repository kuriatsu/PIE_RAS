#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

database = pd.read_csv("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/log/summary.csv")

fig, axes = plt.subplots(1, 3)
sns.countplot(x="int_acc_prob", hue="length", data=database[database.recognition_thresh==0.0], ax=axes[0])
sns.countplot(x="int_acc_prob", hue="length", data=database[database.recognition_thresh==0.5], ax=axes[1])
sns.countplot(x="int_acc_prob", hue="length", data=database[database.recognition_thresh==0.8], ax=axes[2])


cross_total = pd.DataFrame(columns=["thres", "type", "Hit", "Miss", "FA", "CR"])
for thres in [0.0, 0.5, 0.8]:
    buf = pd.Series([
        thres,
        "intervention",
        (database[database.recognition_thresh==thres].int_acc_prob == 0).sum(),
        (database[database.recognition_thresh==thres].int_acc_prob == 1).sum(),
        (database[database.recognition_thresh==thres].int_acc_prob == 2).sum(),
        (database[database.recognition_thresh==thres].int_acc_prob == 3).sum(),
        ], index = cross_total.columns)
    cross_total = cross_total.append(buf, ignore_index=True)

    buf = pd.Series([
        thres,
        "prediction",
        (database[database.recognition_thresh==thres].prediction_acc == 0).sum(),
        (database[database.recognition_thresh==thres].prediction_acc == 1).sum(),
        (database[database.recognition_thresh==thres].prediction_acc == 2).sum(),
        (database[database.recognition_thresh==thres].prediction_acc == 3).sum(),
        ], index = cross_total.columns)
    cross_total = cross_total.append(buf, ignore_index=True)

cross_total_acc_rate = pd.DataFrame(columns=["val", "type", "thres"])
for type in ["intervention", "prediction"]:
    for thres in [0.0, 0.5, 0.8]:
        val = (cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Hit + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].CR) / (cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Hit + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Miss + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].FA + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].CR)
        buf = pd.Series([val, type, thres], index=cross_total_acc_rate.columns)
        cross_total_acc_rate = cross_total_acc_rate.append(buf, ignore_index=True)

cross_length_acc_rate = pd.DataFrame(columns=["val", "type", "thres"])
for thres in [0.0, 0.5, 0.8]:
    val = (cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Hit + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].CR) / (cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Hit + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].Miss + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].FA + cross_total[(cross_total.thres==thres) & (cross_total.type==type)].CR)
    buf = pd.Series([val, type, thres], index=cross_total_acc_rate.columns)
    cross_total_acc_rate = cross_total_acc_rate.append(buf, ignore_index=True)

acc_length = pd.DataFrame(columns=["type", "count", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    for length in database.length.unique():
        for i, type in enumerate(["Hit", "Miss", "FA", "CR"]):
            count = (database[(database.recognition_thresh==thres) & (database.length == length)].int_acc_prob == i).sum()
            buf = pd.Series([type, count, length, thres], index = acc_length.columns)
            acc_length = acc_length.append(buf, ignore_index=True)

sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.0])
sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.5])
sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.8])


intv_count_length = pd.DataFrame(columns=["type", "count", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    for length in database.length.unique():
        count = len(database[(database.recognition_thresh==thres) & (database.length == length) & (database.int_count > 0)])
        buf = pd.Series(["intervention", count, length, thres], index = intv_count_length.columns)
        intv_count_length = intv_count_length.append(buf, ignore_index=True)

        count = len(database[(database.recognition_thresh==thres) & (database.length == length) & ((database.prediction_acc==1) | (database.prediction_acc==2))])
        buf = pd.Series(["necessary", count, length, thres], index = intv_count_length.columns)
        intv_count_length = intv_count_length.append(buf, ignore_index=True)

sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.0])
sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.5])
sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.8])

intv_acc_length = pd.DataFrame(columns=["type", "count", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    for length in database.length.unique():
        for i, type in enumerate(["Hit", "Miss", "FA", "CR"]):
            count = (database[(database.recognition_thresh==thres) & (database.length == length) & (database.int_count > 0)].int_acc_prob == i).sum()
            buf = pd.Series([type, count, length, thres], index = intv_acc_length.columns)
            intv_acc_length = intv_acc_length.append(buf, ignore_index=True)

sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.0])
sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.5])
sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.8])


intention_count_data = pd.DataFrame(columns=["Hit", "Miss", "FA", "CR", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    buf = pd.Series([
        (database[database.recognition_thresh==thres].int_acc_prediction == 0).sum(),
        (database[database.recognition_thresh==thres].int_acc_prediction == 1).sum(),
        (database[database.recognition_thresh==thres].int_acc_prediction == 2).sum(),
        (database[database.recognition_thresh==thres].int_acc_prediction == 3).sum(),
        "total",
        thres,
        ], index = acc_count_data.columns)
    intention_count_data = acc_count_data.append(buf, ignore_index=True)

    for length in database.length.unique():
        buf = pd.Series([
            (database[(database.recognition_thresh==thres) & (database.length == length)].int_acc_prediction == 0).sum(),
            (database[(database.recognition_thresh==thres) & (database.length == length)].int_acc_prediction == 1).sum(),
            (database[(database.recognition_thresh==thres) & (database.length == length)].int_acc_prediction == 2).sum(),
            (database[(database.recognition_thresh==thres) & (database.length == length)].int_acc_prediction == 3).sum(),
            length,
            thres,
            ], index = acc_count_data.columns)
        intention_count_data = acc_count_data.append(buf, ignore_index=True)

int_length_data = pd.DataFrame(columns=[1.0, 3.0, 5.0, 7.0, 9.0, 12.0])
for length in int_length_data.columns:
    index = float(length)
    target = database[database.length==index]
    int_length_data.loc["intervention_total", index] = len(target[((target.result_acc == 0)|(target.result_acc == 3))]) / len(target)
    int_length_data.loc["prediction", index] = len(target[((target.prediction_prob > 0.5) & (target.annt_prob > 0.5)) | ((target.prediction_prob <= 0.5) & (target.annt_prob <= 0.5))]) / len(target)
    int_length_data.loc["total", index] = len(target[((target.last_intention==True) & (target.annt_prob<=0.5)) | ((target.last_intention==False) & (target.annt_prob>0.5))]) / len(target)
