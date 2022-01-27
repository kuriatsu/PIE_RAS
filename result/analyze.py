#! /usr/bin/python3
# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open("/home/kuriatsu/Documents/experiment/pie/202201experiment/database_result_valid.pkl", "rb") as f:
    database = pickle.load(f)

base_dir = "/home/kuriatsu/Documents/experiment/pie/202201experiment/"
result_list = [
    "{}/log_data_takanose_0.0.csv".format(base_dir),
    "{}/log_data_takanose_0.5.csv".format(base_dir),
    "{}/log_data_takanose_0.8.csv".format(base_dir),
    "{}/log_data_tyamamoto_0.0.csv".format(base_dir),
    "{}/log_data_tyamamoto_0.5.csv".format(base_dir),
    "{}/log_data_tyamamoto_0.8.csv".format(base_dir),
]
database.get("3_9_593_12.0").get("int_length")
for i in range(1, len(result_list)):
    result = result.append(pd.read_csv(result_list[i]))
int_length_list = result.int_length.drop_duplicates()
int_length_summary = pd.DataFrame(columns=["int_length", "acc", "int_thresh", "type", "count"])
for i, row in result.iterrows():
    if len(int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh == row.int_thresh)])  == 0:
        for type in ["all", "intervention", "necessary"]:
            buf  = pd.Series([row.int_length, 0, row.int_thresh, type, 0], index=int_length_summary.columns)
            int_length_summary = int_length_summary.append(buf, ignore_index=True)

    data = database.get(row.id + "_" + str(row.int_length))
    if (data.get("prob")>0.5 and row.last_state == True) or (data.get("prob")<=0.5 and row.last_state == False):
        int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="all")]["acc"] += 1
        int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="all")]["count"] += 1

        if not row.int_method.type:
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="intervention")].acc += 1
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="intervention")].count += 1

        if (data.get("prob") > 0.5 and data.get("result") <= 0.5) or data.get("prob") <= 0.5 and data.get("result").get("prob") > 0.5:
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="necessary")].acc += 1
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="necessary")].count += 1
    else:
        int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="all")].count += 1

        if not math.isnan(row.int_method):
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="intervention")].count += 1

        if (data.get("prob") > 0.5 and data.get("result") <= 0.5) or data.get("prob") <= 0.5 and data.get("result").get("prob") > 0.5:
            int_length_summary[(int_length_summary.int_length==row.int_length) & (int_length_summary.int_thresh==row.int_thresh) & (int_length_summary.type=="necessary")].count += 1


    acc = result[(result.int_length==int_length) & ((database.get(result.id).get("prob")>0.5))]

len(int_length_summary[(int_length_summary.int_length==1.0) & (int_length_summary.int_thresh == 0.8)])
