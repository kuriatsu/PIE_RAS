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
    # "/home/kuriatsu/Documents/data/pie_predict/val/result_0-150.pkl",
    # "/home/kuriatsu/Documents/data/pie_predict/val/result_151-243.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/test/result_0-150.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/test/result_151-300.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/test/result_301-450.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/test/result_451-600.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/test/result_601-719.pkl",
    ]
video_list = [
    # "set05/video_0001",
    # "set05/video_0002",
    # "set06/video_0001",
    # "set06/video_0002",
    # "set06/video_0003",
    # "set06/video_0004",
    # "set06/video_0005",
    # "set06/video_0006",
    # "set06/video_0007",
    # "set06/video_0008",
    # "set06/video_0009",
    "set03/video_0001",
    "set03/video_0002",
    "set03/video_0003",
    "set03/video_0004",
    "set03/video_0005",
    "set03/video_0006",
    "set03/video_0007",
    "set03/video_0008",
    "set03/video_0009",
    "set03/video_0010",
    "set03/video_0011",
    "set03/video_0012",
    "set03/video_0013",
    "set03/video_0014",
    "set03/video_0015",
    "set03/video_0016",
    "set03/video_0017",
    "set03/video_0018",
    "set03/video_0019",
]

# load prediction result
data = []
for file in result_file_list:
    with open(file, "rb") as f:
        data+=pickle.load(f)

# get final prediction result for each pedestrian
result_dict = pd.DataFrame({"prob":[],"result":[]})
for buf in data:
    if buf.get("ped_id") not in result_dict.index:
        series = pd.Series([0,float(buf.get("res"))], index=result_dict.columns, name=buf.get("ped_id"))
        result_dict = result_dict.append(series)
    else:
        result_dict.at[buf.get("ped_id"), "result"] = buf.get("res")

# get annotation
annotation_len_list = []
exp_len_list = []
annotation_len_list_tl = []
for video in video_list:
    annt = getXmlRoot("/home/kuriatsu/Documents/data/annotations/{}_annt.xml".format(video))
    attrib = getXmlRoot("/home/kuriatsu/Documents/data/annotations_attributes/{}_attributes.xml".format(video))

    for track in annt.iter("track"):
        if track.attrib.get("label") == "pedestrian":
            for attribute in track[0].iter("attribute"):
                if attribute.get("name") == "id":
                    ped_id = attribute.text
                    break

            for pedestrian in attrib.iter("pedestrian"):
                if pedestrian.get("id") == ped_id:
                    exp_len_list.append((int(pedestrian.attrib.get("critical_point"))-int(pedestrian.attrib.get("exp_start_point")))/30)
                    result_dict.at[pedestrian.attrib.get("id"), "prob"] = float(pedestrian.attrib.get("intention_prob"))
                    annotation_len_list.append((int(pedestrian.attrib.get("critical_point")) - int(track[0].attrib.get("frame")))/30)
                    break

        if track.attrib.get("label") == "traffic_light":
            annotation_len_list_tl.append((int(track[-1].attrib.get("frame")) - int(track[0].attrib.get("frame")))/30)

            break
    del annt
    del attrib

# visualize annotation length
fig, axes = plt.subplots()
sns.histplot(annotation_len_list, binwidth=1, color="tomato", ax=axes, label="max_int_time")
sns.histplot(exp_len_list, binwidth=1, color="steelblue", ax=axes, label="exp_time")
sns.histplot(annotation_len_list_tl, binwidth=1, color="seagreen", ax=axes, label="max_int_time(tl)")
plt.legend()
plt.show()

# visualize prediction result
melted_df = result_dict.dropna().melt(value_vars=["prob", "result"], var_name="type", value_name="value")
sns.histplot(data=melted_df, binwidth=0.05, x="value", hue="type")
plt.show()

# calc accuracy
fp_list = {}
fn_list = {}

for thres in range(0, 11):
    fp_count = 0
    fn_count = 0
    for index, result in result_dict.iterrows():
        fn_count+=(result["prob"] >= thres*0.1 and result["result"] < thres*0.1)
        fp_count+=(result["prob"] < thres*0.1 and result["result"] >= thres*0.1)
    fp_list[thres*0.1] = fp_count/len(result_dict.index)
    fn_list[thres*0.1] = fn_count/len(result_dict.index)

fig, axes = plt.subplots()
sns.lineplot(x=fp_list.keys(), y=fp_list.values(), color="tomato", label="false-positive", ax=axes)
sns.lineplot(x=fn_list.keys(), y=fn_list.values(), color="steelblue", label="false-negative", ax=axes)
plt.legend()
plt.show()
