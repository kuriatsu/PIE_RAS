#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle
import numpy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

def getXmlRoot(filename):
    tree = ET.parse(filename)
    return tree.getroot()


result_file_list = [
    # "/home/kuriatsu/Documents/data/pie_predict/val/result_0-150.pkl",
    # "/home/kuriatsu/Documents/data/pie_predict/val/result_151-243.pkl",
    "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_0-150.pkl",
    "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_150-300.pkl",
    "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_300-450.pkl",
    "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_450-600.pkl",
    "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_600-719.pkl",
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

attrib_dict = {}
for video in video_list:
    attrib = getXmlRoot("/media/kuriatsu/SamsungKURI/PIE_data/annotations_attributes/{}_attributes.xml".format(video))
    for pedestrian in attrib.iter("pedestrian"):
        attrib_dict[pedestrian.get("id")] = {
            "critical_point": pedestrian.get("critical_point"),
            "intention_prob": pedestrian.get("intention_prob"),
            }

# load prediction result
data = []
for file in result_file_list:
    with open(file, "rb") as f:
        data+=pickle.load(f)

# and add intention_prob
for buf in data:
    buf["intention_prob"] = attrib_dict.get(buf.get("ped_id")).get("intention_prob")

# summarize prediction result and GT
result_dict = pd.DataFrame({"prob":[],"result":[], "target_frame":[]})
for buf in data:
    target_frame = int(buf.get("imp").split("/")[-1].replace(".png", ""))
    video = buf.get("imp").split("/")[-3] + "/" + buf.get("imp").split("/")[-2]
    attrib = attrib_dict.get(buf.get("ped_id"))

    if int(attrib.get("critical_point")) == target_frame:
        series = pd.Series([float(attrib.get("intention_prob")), float(buf.get("res")), target_frame], index=result_dict.columns, name=buf.get("ped_id"))
        result_dict = result_dict.append(series)

# visualize prediction result
melted_df = result_dict.dropna().melt(value_vars=["prob", "result"], var_name="type", value_name="value")
sns.histplot(data=melted_df, binwidth=0.05, x="value", hue="type")
plt.show()

# calc accuracy
tp_list = {}
tn_list = {}
fp_list = {}
fn_list = {}
precision = {}
recall = {}
for thres in range(0, 11):
    tp = tn = fp = fn = 0
    for index, result in result_dict.iterrows():
        if math.isnan(result["result"]):
            continue
        tp+=(result["prob"] > 0.5 and result["result"] > thres*0.1)
        tn+=(result["prob"] <= 0.5 and result["result"] <= thres*0.1)
        fp+=(result["prob"] <= 0.5 and result["result"] > thres*0.1)
        fn+=(result["prob"] > 0.5 and result["result"] <= thres*0.1)
    tp_list[thres*0.1] = tp/len(result_dict.index)
    tn_list[thres*0.1] = tn/len(result_dict.index)
    fp_list[thres*0.1] = fp/len(result_dict.index)
    fn_list[thres*0.1] = fn/len(result_dict.index)

    if tp == 0:
        precision[thres*0.1] = 0.0
        recall[thres*0.1] = 0.0
    else:
        precision[thres*0.1] = tp/(tp+fp)
        recall[thres*0.1] = tp/(tp+fn)


    print((tp+tn)/(tp+tn+fn+fp))


fig, axes = plt.subplots()
sns.lineplot(x=fp_list.keys(), y=fp_list.values(), color="lightgreen", label="false-positive", ax=axes)
sns.lineplot(x=fn_list.keys(), y=fn_list.values(), color="violet", label="false-negative", ax=axes)
sns.lineplot(x=tp_list.keys(), y=tp_list.values(), color="tomato", label="true-positive", ax=axes)
sns.lineplot(x=tn_list.keys(), y=tn_list.values(), color="steelblue", label="true-negative", ax=axes)
plt.legend()
plt.show()

fig, axes = plt.subplots()
sns.lineplot(x=recall.values(), y=precision.values(), color="lightgreen", label="false-positive", ax=axes)
plt.legend()
plt.show()



#####################################
# check flame length get annotation
#####################################
annotation_len_list = []
exp_len_list = []
annotation_len_list_tl = []
for video in video_list:
    annt = getXmlRoot("/media/kuriatsu/SamsungKURI/PIE_data/annotations/{}_annt.xml".format(video))
    attrib = getXmlRoot("/media/kuriatsu/SamsungKURI/PIE_data/annotations_attributes/{}_attributes.xml".format(video))

    for track in annt.iter("track"):
        if track.attrib.get("label") == "pedestrian":
            for attribute in track[0].iter("attribute"):
                if attribute.get("name") == "id":
                    ped_id = attribute.text
                    break

            for pedestrian in attrib.iter("pedestrian"):
                if pedestrian.get("id") == ped_id:
                    exp_len_list.append((int(pedestrian.attrib.get("critical_point"))-int(pedestrian.attrib.get("exp_start_point")))/30)
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


#####################################
# analyze extracted data accuracy distribution
#####################################

with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid.pkl", 'rb') as f:
    database = pickle.load(f)

# calc accuracy
tp_list = {}
tn_list = {}
fp_list = {}
fn_list = {}
precision = {}
recall = {}
for thres in range(0, 11):
    tp = tn = fp = fn = 0
    for index, data in database.items():
        if math.isnan(result["result"]):
            continue
        tp+=(data["prob"] > 0.5 and data["results"] > thres*0.1)
        tn+=(data["prob"] <= 0.5 and data["results"] <= thres*0.1)
        fp+=(data["prob"] <= 0.5 and data["results"] > thres*0.1)
        fn+=(data["prob"] > 0.5 and data["results"] <= thres*0.1)
    tp_list[thres*0.1] = tp/len(database)
    tn_list[thres*0.1] = tn/len(database)
    fp_list[thres*0.1] = fp/len(database)
    fn_list[thres*0.1] = fn/len(database)

    if tp == 0:
        precision[thres*0.1] = 0.0
        recall[thres*0.1] = 0.0
    else:
        precision[thres*0.1] = tp/(tp+fp)
        recall[thres*0.1] = tp/(tp+fn)


    print((tp+tn)/(tp+tn+fn+fp))


fig, axes = plt.subplots()
sns.lineplot(x=fp_list.keys(), y=fp_list.values(), color="lightgreen", label="false-positive", ax=axes)
sns.lineplot(x=fn_list.keys(), y=fn_list.values(), color="violet", label="false-negative", ax=axes)
sns.lineplot(x=tp_list.keys(), y=tp_list.values(), color="tomato", label="true-positive", ax=axes)
sns.lineplot(x=tn_list.keys(), y=tn_list.values(), color="steelblue", label="true-negative", ax=axes)
plt.legend()
plt.show()
