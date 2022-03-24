#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import random
from multiprocessing import Pool, Manager
# import threading

def getVideo(filename):

    try:
        video = cv2.VideoCapture(filename)

    except:
        print('cannot open video')
        exit(0)

    # get video rate and change variable unit from time to frame num
    fps = int(video.get(cv2.CAP_PROP_FPS))
    image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]

    return video, image_res, fps


def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()


def getAnntAtrrib(root, id):
    for pedestrian in root.iter("pedestrian"):
        if pedestrian.attrib.get("id") == id:
            return pedestrian

    return None


def getAtrrib(root, tag, attrib, attrib_target):
    for i in root.iter(tag):
        if i.attrib.get(attrib) == attrib_target:
            return i.text

    return None

def getVehicleDirection(vehicle_root, start_frame, end_frame):
    dist_buf = 0
    # for i in range(start_frame, len(vehicle_root)):
    #     dist_buf += float(vehicle_root[i].get('OBD_speed')) * 0.03 / 3.6
    #     future_angle = float(vehicle_root[end_frame].get('yaw')) - float(vehicle_root[i].get('yaw'))
    #     if dist_buf > 50:
    #         break

    # get frame 10m before the end_frame
    mileage = 0
    target_frame = 0
    for i in range(end_frame, len(vehicle_root)):
        mileage += float(vehicle_root[i].get("OBD_speed"))/(3.6*30)
        if mileage > 10:
            target_frame = i
            break
    if target_frame == 0:
        target_frame = len(vehicle_root)-1

    # get outer prod between 10m after end_frame and start_frame+0.1sec
    angle_diff = float(vehicle_root[target_frame].get("yaw")) - float(vehicle_root[start_frame+10].get("yaw"))
    prod = np.sin(angle_diff)
    if 0.4 > abs(prod):
        return 'straight'
    else:
        if prod > 0.0:
            return 'right'
        else:
            return 'left'


def getVehicleBrakeFrame(vehicle_root, start_frame, crossing_point):
    distance = 0
    g = 2.0
    sefety_mergin = 10
    for frame in range(crossing_point, start_frame, -1):
        speed = float(vehicle_root[frame].get("OBD_speed"))
        distance += speed/(3.6*30)
        stop_distance = (speed**2) / (2*g*9.8) + safety_mergin
        if stop_distance < distance:
            return frame

def getAnchor(track, start_frame, end_frame):
    anchor_list = []
    for frame in range(start_frame, end_frame+1):
        is_found = False
        for box in track.iter("box"):
            if int(box.get("frame")) == frame:
                anchor = {
                    "xbr": int(float(box.get('xbr'))),
                    "xtl": int(float(box.get('xtl'))),
                    "ybr": int(float(box.get('ybr'))),
                    "ytl": int(float(box.get('ytl'))),
                    }
                anchor_list.append(anchor)
                is_found = True
                break
        if not is_found:
            anchor_list.append({})

    return anchor_list


def process(database, video_name):
    max_after_length = 2*30 # second x frame_rate
    int_length_list = [1.0, 3.0, 5.0, 8.0]
    tl_state_list = {"red":1, "green":0}

    annt_attribute_root = getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_name))
    annt_root = getXmlRoot("{}/annotations/{}_annt.xml".format(base_dir, video_name))
    ego_vehicle_root = getXmlRoot("{}/annotations_vehicle/{}_obd.xml".format(base_dir, video_name))
    video_file = "{}/PIE_clips/{}.mp4".format(base_dir, video_name)
    _, image_res, _ = getVideo(video_file)

    for track in annt_root.iter("track"):
        if track.get("label") == "pedestrian":
            ped_id = getAtrrib(track[0], "attribute", "name", "id")
            ped_attrib = getAnntAtrrib(annt_attribute_root, ped_id)
            max_len = (int(ped_attrib.get("critical_point")) - int(track[0].get("frame")))/30

            # remove short video or pedestrians who cross non relevant road
            if max_len < min(int_length_list) or ped_attrib.get("crossing") == "-1":
                continue

            # cut video each size
            for int_length in int_length_list:
                # remove short video
                if int_length > max_len:
                    break

                start_frame_int = int(int(ped_attrib.get("critical_point")) - int_length * 30)
                end_frame = int(track[-1].get("frame"))

                # extract pie prediction result
                if result_dict.get(ped_id).get(start_frame_int) is None:
                    prediction = result_dict.get(ped_id).get(min(result_dict.get(ped_id).keys()))
                else:
                    prediction = result_dict.get(ped_id).get(start_frame_int)

                video_database = {
                    "video_file" : video_file,
                    "id" : ped_id,
                    "label" : "int",
                    "int_length" : int_length,
                    "likelihood" : prediction,
                    "anchor" : getAnchor(track, start_frame_int, end_frame),
                    "future_direction" : getVehicleDirection(ego_vehicle_root, start_frame_int, end_frame),
                    "start_frame" : start_frame_int,
                    "critical_point" :float(ped_attrib.get("critical_point")),
                    "end_frame" :  min(int(ped_attrib.get("critical_point"))+max_after_length, end_frame),
                    "state" : float(ped_attrib.get("intention_prob")),
                    }
                name = "{}int_{}".format(ped_id, int_length)
                database[name] = video_database

                traj_critical_point = None
                if ped_attrib.get("crossing") == "1":
                    traj_critical_point = int(ped_attrib.get("crossing_point"))
                elif ped_attrib.get("crossing") == "0":
                    traj_critical_point = getVehicleBrakeFrame(ego_vehicle_root, int(track[0].get("frame")), int(ped_attrib.get("crossing_point")))

                start_frame_traj = int(traj_critical_point - int_length * 30)

                video_database = {
                    "video_file" : video_file,
                    "id" : ped_id,
                    "label" : "traj",
                    "int_length" : int_length,
                    "likelihood" : prediction,
                    "anchor" : getAnchor(track, start_frame_traj, end_frame),
                    "future_direction" :  getVehicleDirection(ego_vehicle_root, start_frame_traj, end_frame),
                    "start_frame" :  start_frame_traj,
                    "critical_point" : traj_critical_point,
                    "end_frame" :  min(int(ped_attrib.get("crossing_point"))+max_after_length, end_frame),
                    "state" :  ped_attrib.get("crossing"),
                    }
                name = "{}traj_{}".format(ped_id, int_length)
                database[name] = video_database


        elif track.get("label") == "traffic_light":
            tl_id = getAtrrib(track[0], "attribute", "name", "id")
            type = getAtrrib(track[0], "attribute", "name", "type")

            # skip non reqular light or contraflow light
            # if type != "regular" or float(track[0].get("xtl")) < image_res[1]*0.5:
            if type != "regular":
                continue

            max_len = (int(track[-1].get("frame")) - int(track[0].get("frame")))/30

            # short video
            if max_len < min(int_length_list):
                continue

            # cut video each size
            for int_length in int_length_list:
                if int_length > max_len:
                    break

                start_frame = int(int(track[-1].get("frame")) - int_length * 30)
                end_frame = int(track[-1].get("frame"))
                print(name, start_frame, end_frame)

                video_database = {
                    "video_file" : video_file,
                    "id" : tl_id,
                    "label" : "traffic_light",
                    "int_length" : int_length,
                    "likelihood" : random.random(),
                    "anchor" :  getAnchor(track, start_frame, end_frame),
                    "future_direction" :  getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                    "start_frame" : start_frame,
                    "critical_point" : int(track[-1].get('frame')),
                    "end_frame" :  int(track[-1].get('frame')),
                    "state" : tl_state_list.get(getAtrrib(track[-1], "attribute", "name", "state")),
                    }

                name = "{}_{}".format(tl_id, int_length)
                database[name] = video_database


base_dir = "./data/PIE_data"

result_file_list = [
    base_dir + "/pie_predict/test/result_0-150.pkl",
    base_dir + "/pie_predict/test/result_150-300.pkl",
    base_dir + "/pie_predict/test/result_300-450.pkl",
    base_dir + "/pie_predict/test/result_450-600.pkl",
    base_dir + "/pie_predict/test/result_600-719.pkl",
    # base_dir + "/pie_predict/val/result_0-150.pkl",
    # base_dir + "/pie_predict/val/result_151-243.pkl",
    ]

# load prediction result
prediction_data = []
for file in result_file_list:
    with open(file, "rb") as f:
        prediction_data+=pickle.load(f)

prediction_data[0]
# get final prediction result for each pedestrian
result_dict = {}
for buf in prediction_data:
    frame = int(buf.get("imp").split("/")[-1].replace(".png", ""))
    if result_dict.get(buf.get("ped_id")) is None:
        result_dict[buf.get("ped_id")] = {frame : float(buf.get("res"))}
    else:
        result_dict[buf.get("ped_id")][frame] = float(buf.get("res"))

set_list = ["set03"]

video_list = [
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
    # "set05/video_0001",
    # "set05/video_0002",
]

## multi processing
with Manager() as manager:
    p = Pool(4)
    database = manager.dict()
    # database = {}
    for video in video_list:
        p.apply_async(process, args=(database, video))

    p.close()
    p.join()
    print(dict(database).keys())

    with open("{}/experiment/database.pkl".format(base_dir), "wb") as f:
        pickle.dump(dict(database), f)



## single process
# database = {}
# for video in video_list:
#     process(database, video)
#
# with open("{}/extracted_data/database.pkl".format(base_dir), "wb") as f:
#     pickle.dump(database, f)
