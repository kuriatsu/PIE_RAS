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
data_path = "/home/kuriatsu/Dropbox/data/pie_experiment202201"
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

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid.pkl", "rb") as f:
    database_valid = pickle.load(f)

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database.pkl", "rb") as f:
    database = pickle.load(f)

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
                                "result_acc", # 0=hit, 1=miss, 2=FA, 3=CR
                                "result_intention",
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
    if data_id not in database_valid.keys() and data_id not in database.keys():
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
            buf.result_acc = 3
        else:
            buf.result_acc = 2

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.result_acc = 3
        else:
            buf.result_acc = 2

    elif buf.prediction_prob > buf.recognition_thresh and buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.result_acc = 0
        else:
            buf.result_acc = 1

    elif buf.prediction_prob <= buf.recognition_thresh and buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.result_acc = 0
        else:
            buf.result_acc = 1

    if buf.annt_prob <= 0.5:
        if buf.last_intention:
            buf.result_intention = 3
        else:
            buf.result_intention = 2

    elif  buf.annt_prob > 0.5:
        if not buf.last_intention:
            buf.result_intention = 0
        else:
            buf.result_intention = 1

    extracted_data = extracted_data.append(buf, ignore_index=True)


length_acc = pd.DataFrame(columns=[1.0, 3.0, 5.0, 7.0, 9.0, 12.0])
for length in length_acc.columns:
    index = float(length)
    target = extracted_data[extracted_data.length==index]
    length_acc.loc["intervention_total", index] = len(target[((target.result_acc == 0)|(target.result_acc == 3))]) / len(target)
    length_acc.loc["prediction", index] = len(target[((target.prediction_prob > 0.5) & (target.annt_prob > 0.5)) | ((target.prediction_prob <= 0.5) & (target.annt_prob <= 0.5))]) / len(target)
    length_acc.loc["total", index] = len(target[((target.last_intention==True) & (target.annt_prob<=0.5)) | ((target.last_intention==False) & (target.annt_prob>0.5))]) / len(target)
