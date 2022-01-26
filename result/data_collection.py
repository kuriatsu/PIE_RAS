#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import glob
import os

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

with open("/home/kuriatsu/Dropbox/data/pie_predict/database_result_valid.pkl", "rb") as f:
    database_valid = pickle.load(f)

with open("/home/kuriatsu/Dropbox/data/pie_predict/database.pkl", "rb") as f:
    database = pickle.load(f)

extracted_data = pd.DataFrame(columns=[
                                "subject",
                                "trial"
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
                                "crossed",
                                ])

for i, row in log_data.iterrows():

    data_id = row.id+"_"+str(row.int_length)
    buf = pd.Series([
        row.subject,
        row.trial,
        row.id,
        row.int_thresh,
        row.int_length,
        row.first_int_time,
        row.last_int_time,
        row.int_count,
        database_valid.get(data_id).get("intention_prob"),
        database_valid.get(data_id).get("res"),
        "cross" if row.last_state else "no_cross",
        None,
        ], index=extracted_data.columns)

    if row.subject in ["tyamamoto", "takanose"]:
        buf.prediction_res = database.get(data_id).get("res")
