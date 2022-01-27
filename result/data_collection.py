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
                                "intention_prob",
                                "prediction_prob",
                                "last_crossing_intention",
                                "int_result",
                                "crossed"
                                ])

for i, row in log_data.iterrows():

    if type(row.id) is not str and math.isnan(row.id):
        continue

    data_id = row.id+"_"+str(row.int_length)
    if not data_id.endswith(".0"):
        data_id = data_id+".0"
    if data_id not in database_valid.keys() and data_id not in database.keys():
        print(f"{data_id} not found in {row.subject}, {row.trial}")
        continue

    buf = pd.Series([
        row.subject,
        row.trial,
        row.id,
        row.int_thresh,
        row.int_length,
        row.first_int_time,
        row.last_int_time,
        row.int_count,
        None,
        None,
        "cross" if row.last_state else "no_cross",
        None,
        None,
        ], index=extracted_data.columns)

    # if row.subject in ["tyamamoto", "takanose"]:
    #     buf["intention_prob"] = database.get(data_id).get("prob")
    #     buf["prediction_res"] = database.get(data_id).get("results")
    # else:
    buf["intention_prob"]= database_valid.get(data_id).get("prob")
    buf["prediction_prob"] = database_valid.get(data_id).get("results")

    extracted_data = extracted_data.append(buf, ignore_index=True)


for i, row in extracted_data.iterrows():
    if row.prediction_prob is None or row.prediction_prob is None:
        print(row)
