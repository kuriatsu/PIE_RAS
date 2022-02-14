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
data_path = "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/log"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    buf = pd.read_csv(file)
    filename =file.split("/")[-1]
    subject = filename.rsplit("_", 1)[0].replace("log_data_", "")
    if subject in ["kanayama"]:
        break
    trial = filename.split("_")[-1].replace(".csv", "")
    buf["subject"] = subject
    buf["trial"] = trial
    if log_data is None:
        log_data = buf
    else:
        log_data = log_data.append(buf, ignore_index=True)

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid_cross.pkl", "rb") as f:
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
                                "int_acc_prediction", # 0=hit, 1=miss, 2=FA, 3=CR
                                "int_acc_prob",
                                "prediction_acc",
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
        int(database_valid.get(data_id).get("crossing")),
        None,
        None,
        None,
        ], index=extracted_data.columns)
    # print(database_valid.get(data_id).get("crossing"))
    # if row.subject in ["tyamamoto", "takanose"]:
    #     buf["intention_prob"] = database.get(data_id).get("prob")
    #     buf["prediction_res"] = database.get(data_id).get("results")
    # else:
    #     buf["intention_prob"]= database_valid.get(data_id).get("prob")
    #     buf["prediction_prob"] = database_valid.get(data_id).get("results")
    #     buf["crossing"] = database_valid.get(data_id).get("crossing")

    if buf.prediction_prob > buf.recognition_thresh and buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_acc_prediction = 3
        else:
            buf.int_acc_prediction = 2

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_acc_prediction = 3
        else:
            buf.int_acc_prediction = 2

    elif buf.prediction_prob > buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_acc_prediction = 0
        else:
            buf.int_acc_prediction = 1

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_acc_prediction = 0
        else:
            buf.int_acc_prediction = 1

    if buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.int_acc_prob = 3
        else:
            buf.int_acc_prob = 2

    elif  buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.int_acc_prob = 0
        else:
            buf.int_acc_prob = 1

    if buf.annt_prob <= 0.5:
        if buf.prediction_prob <= buf.recognition_thresh:
            buf.prediction_acc = 3
        else:
            buf.prediction_acc = 2

    elif  buf.annt_prob > 0.5:
        if buf.prediction_prob > buf.recognition_thresh:
            buf.prediction_acc = 0
        else:
            buf.prediction_acc = 1

    extracted_data = extracted_data.append(buf, ignore_index=True)

extracted_data.to_csv(data_path+"/summary2.csv")
