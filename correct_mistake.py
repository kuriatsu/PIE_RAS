#! /usr/bin/python3
# -*- coding:utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import glob
import os
import math
import xml.etree.ElementTree as ET
import cv2
from pie_ras import PIERas

import matplotlib.pyplot as plt
import seaborn as sns



miss_data = []
false_data = []
data_path = "/home/kuriatsu/Dropbox/data/pie202203"
for file in glob.glob(os.path.join(data_path, "log*.csv")):
    buf = pd.read_csv(file)
    filename =file.split("/")[-1]
    task = filename.replace("log_data_", "").split("_")[1]
    count = int(filename.replace("log_data_", "").split("_")[-1].replace(".csv", ""))
    if count in [0, 1, 2]:
        print("{} skipped".format(filename))
        continue
    for idx, row in buf.iterrows():
        if task == "tl":
            id = f"{row.id}_{row.int_length}"
        else:
            id = f"{row.id}{task}_{row.int_length}"
        if row.last_state == -1:
            miss_data.append(id)
        elif row.state == row.last_state:
            false_data.append(id)

miss_data = list(set(miss_data))
false_data = list(set(false_data))
with open("./data/PIE_data/experiment/database.pkl", 'rb') as f:
    database = pickle.load(f)

with PIERas("traj") as pie_ras:
    for id in false_data:
        if id.rsplit("_", 1)[0].endswith("traj"):
            pie_ras.play(database.get(id), "result")
            # try:
            # except KeyboardInterrupt:
            #     break
            # except Exception:
            #     pass
