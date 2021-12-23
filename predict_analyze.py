#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def getXmlRoot(filename):
    tree = ET.parse(filename)
    return tree.getroot()


result_file_list = [
    "/home/kuriatsu/Documents/data/pie_predict/result_0-150.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_151-300.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_301-450.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_451-600.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_601-719.pkl",
    ]
video_list = [
    "video_0001",
    "video_0002",
    "video_0003",
    "video_0004",
    "video_0005",
    "video_0006",
    "video_0007",
    "video_0008",
    "video_0009",
    "video_0010",
    "video_0011",
    "video_0012",
    "video_0013",
    "video_0014",
    "video_0015",
    "video_0016",
    "video_0017",
    "video_0018",
    "video_0019",
]

data = []
for file in result_file_list:
    with open(file, "rb") as f:
        data+=pickle.load(f)

result_dict = pd.DataFrame({"prob":[],"result":[]})
for buf in data:
    if buf.get("ped_id") not in result_dict.index:
        series = pd.Series([0,float(buf.get("res"))], index=result_dict.columns, name=buf.get("ped_id"))
        result_dict = result_dict.append(series)
    else:
        result_dict.at[buf.get("ped_id"), "result"] = buf.get("res")

annotation_len_list = []
exp_len_list = []

for video in video_list:
    annt = getXmlRoot("/home/kuriatsu/Documents/data/annotations/{}/{}_annt.xml".format("set03", video))
    attrib = getXmlRoot("/home/kuriatsu/Documents/data/annotations_attributes/{}/{}_attributes.xml".format("set03", video))

    for track in annt.iter("track"):
        if track.attrib.get("label") == "pedestrian":
            annotation_len_list.append((int(track[-1].attrib.get("frame")) - int(track[0].attrib.get("frame")))/30)

    for pedestrian in attrib.iter("pedestrian"):
        exp_len_list.append((int(pedestrian.attrib.get("critical_point"))-int(pedestrian.attrib.get("exp_start_point")))/30)
        result_dict.at[pedestrian.attrib.get("id"), "prob"] = float(pedestrian.attrib.get("intention_prob"))

    del annt
    del attrib

fig, axes = plt.subplots()
sns.histplot(annotation_len_list, binwidth=1, color="tomato", ax=axes)
sns.histplot(exp_len_list, binwidth=1, color="steelblue", ax=axes)
plt.show()

melted_df = result_dict.dropna().melt(value_vars=["prob", "result"], var_name="type", value_name="value")
sns.histplot(data=melted_df, binwidth=0.05, x="value", hue="type")
plt.show()

correct_count = 0
for result in result_dict.iterrows():
    correct_count+=(result[1]["prob"] < 0.5 and result[1]["result"] < 0.5) or (result[1]["prob"] >= 0.5 and result[1]["result"] >= 0.5)
acc = correct_count/len(result_dict.index)
