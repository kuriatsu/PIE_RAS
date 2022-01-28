#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import glob
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns


# get data
log_data = None
data_path = "/home/kuriatsu/Documents/experiment/pie_202201"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    buf = pd.read_csv(file)
    filename =file.split("/")[-1]
    subject = filename.rsplit("_", 1)[0].replace("log_data_", "")
    trial = filename.split("_")[-1].replace(".csv", "")
    buf["subject"] = subject
    buf["trial"] = trial
    if log_data is None:
        log_data = buf
    else:
        log_data = log_data.append(buf, ignore_index=True)

with open("/home/kuriatsu/Documents/experiment/pie_202201/database_result_valid_cross.pkl", "rb") as f:
    database_valid = pickle.load(f)

# with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database.pkl", "rb") as f:
#     database = pickle.load(f)

extracted_data = pd.DataFrame(columns=[
                                "subject",
                                "trial",
                                "id",
                                "recognition_thresh", # 0.0, 0.5, 0.8 thresh of recognition system
                                "length",             # time limit for intervention
                                "first_int_time",
                                "last_int_time",
                                "int_count",
                                "annt_prob",
                                "prediction_prob",
                                "last_intention", # False=cross True=no_cross
                                "crossing",
                                "int_acc", # 0=hit, 1=miss, 2=FA, 3=CR
                                "int_cross_acc",
                                "prediction_cross_acc",
                                ])

for i, row in log_data.iterrows():

    # avoid nan values
    if type(row.id) is not str and math.isnan(row.id):
        continue

    # create id from ped_id + int_length
    data_id = row.id+"_"+str(row.int_length)
    if not data_id.endswith(".0"):
        data_id = data_id+".0"

    # no id in database
    if data_id not in database_valid.keys():
    # if data_id not in database_valid.keys() and data_id not in database.keys():
        print(f"{data_id} not found in {row.subject}, {row.trial}")
        continue

    buf = pd.Series([
        row.subject,
        int(row.trial),
        row.id,
        float(row.int_thresh),
        float(row.int_length),
        float(row.first_int_time),
        float(row.last_int_time),
        int(row.int_count),
        float(database_valid.get(data_id).get("prob")),
        float(database_valid.get(data_id).get("results")),
        row.last_state,
        None, # database_valid.get(data_id).get("crossing"),
        None,
        None,
        None,
        ], index=extracted_data.columns)

    # if row.subject in ["tyamamoto", "takanose"]:
    #     buf["intention_prob"] = database.get(data_id).get("prob")
    #     buf["prediction_res"] = database.get(data_id).get("results")
    # else:
    #     buf["intention_prob"]= database_valid.get(data_id).get("prob")
    #     buf["prediction_prob"] = database_valid.get(data_id).get("results")
    #     buf["crossing"] = database_valid.get(data_id).get("crossing")

    if buf.prediction_prob > buf.recognition_thresh and buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_acc = 3
        else:
            buf.int_acc = 2

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_acc = 3
        else:
            buf.int_acc = 2

    elif buf.prediction_prob > buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_acc = 0
        else:
            buf.int_acc = 1

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_acc = 0
        else:
            buf.int_acc = 1

    if buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_cross_acc = 3
        else:
            buf.int_cross_acc = 2

    elif  buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_cross_acc = 0
        else:
            buf.int_cross_acc = 1

    if buf.annt_prob <= 0.5:
        if buf.prediction_prob <= buf.recognition_thresh:
            buf.prediction_cross_acc = 3
        else:
            buf.prediction_cross_acc = 2

    elif  buf.annt_prob > 0.5:
        if buf.prediction_prob > buf.recognition_thresh:
            buf.prediction_cross_acc = 0
        else:
            buf.prediction_cross_acc = 1

    extracted_data = extracted_data.append(buf, ignore_index=True)


cross_total = pd.DataFrame(columns=["thres", "type", "Hit", "Miss", "FA", "CR"])
for thres in [0.0, 0.5, 0.8]:
    buf = pd.Series([
        thres,
        "intervention",
        (extracted_data[extracted_data.recognition_thresh==thres].int_cross_acc == 0).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_cross_acc == 1).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_cross_acc == 2).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_cross_acc == 3).sum(),
        ], index = cross_total.columns)
    cross_total = cross_total.append(buf, ignore_index=True)

    buf = pd.Series([
        thres,
        "prediction",
        (extracted_data[extracted_data.recognition_thresh==thres].prediction_cross_acc == 0).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].prediction_cross_acc == 1).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].prediction_cross_acc == 2).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].prediction_cross_acc == 3).sum(),
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
    for length in extracted_data.length.unique():
        for i, type in enumerate(["Hit", "Miss", "FA", "CR"]):
            count = (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length)].int_cross_acc == i).sum()
            buf = pd.Series([type, count, length, thres], index = acc_length.columns)
            acc_length = acc_length.append(buf, ignore_index=True)

sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.0])
sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.5])
sns.barplot(x="type", y="count", hue="length", data=acc_length[acc_length.thres == 0.8])


intv_count_length = pd.DataFrame(columns=["type", "count", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    for length in extracted_data.length.unique():
        count = len(extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length) & (extracted_data.int_count > 0)])
        buf = pd.Series(["intervention", count, length, thres], index = intv_count_length.columns)
        intv_count_length = intv_count_length.append(buf, ignore_index=True)

        count = len(extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length) & ((extracted_data.prediction_cross_acc==1) | (extracted_data.prediction_cross_acc==2))])
        buf = pd.Series(["necessary", count, length, thres], index = intv_count_length.columns)
        intv_count_length = intv_count_length.append(buf, ignore_index=True)

sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.0])
sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.5])
sns.barplot(x="length", y="count", hue="type", data=intv_count_length[intv_count_length.thres == 0.8])

intv_acc_length = pd.DataFrame(columns=["type", "count", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    for length in extracted_data.length.unique():
        for i, type in enumerate(["Hit", "Miss", "FA", "CR"]):
            count = (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length) & (extracted_data.int_count > 0)].int_cross_acc == i).sum()
            buf = pd.Series([type, count, length, thres], index = intv_acc_length.columns)
            intv_acc_length = intv_acc_length.append(buf, ignore_index=True)

sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.0])
sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.5])
sns.barplot(x="type", y="count", hue="length", data=intv_acc_length[intv_acc_length.thres == 0.8])


intention_count_data = pd.DataFrame(columns=["Hit", "Miss", "FA", "CR", "length", "thres"])
for thres in [0.0, 0.5, 0.8]:
    buf = pd.Series([
        (extracted_data[extracted_data.recognition_thresh==thres].int_acc == 0).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_acc == 1).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_acc == 2).sum(),
        (extracted_data[extracted_data.recognition_thresh==thres].int_acc == 3).sum(),
        "total",
        thres,
        ], index = acc_count_data.columns)
    intention_count_data = acc_count_data.append(buf, ignore_index=True)

    for length in extracted_data.length.unique():
        buf = pd.Series([
            (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length)].int_acc == 0).sum(),
            (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length)].int_acc == 1).sum(),
            (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length)].int_acc == 2).sum(),
            (extracted_data[(extracted_data.recognition_thresh==thres) & (extracted_data.length == length)].int_acc == 3).sum(),
            length,
            thres,
            ], index = acc_count_data.columns)
        intention_count_data = acc_count_data.append(buf, ignore_index=True)

int_length_data = pd.DataFrame(columns=[1.0, 3.0, 5.0, 7.0, 9.0, 12.0])
for length in int_length_data.columns:
    index = float(length)
    target = extracted_data[extracted_data.length==index]
    int_length_data.loc["intervention_total", index] = len(target[((target.result_acc == 0)|(target.result_acc == 3))]) / len(target)
    int_length_data.loc["prediction", index] = len(target[((target.prediction_prob > 0.5) & (target.annt_prob > 0.5)) | ((target.prediction_prob <= 0.5) & (target.annt_prob <= 0.5))]) / len(target)
    int_length_data.loc["total", index] = len(target[((target.last_intention==True) & (target.annt_prob<=0.5)) | ((target.last_intention==False) & (target.annt_prob>0.5))]) / len(target)
