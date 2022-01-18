#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import random
from multiprocessing import Pool, Manager
# import threading

def getVideo(filename, image_offset_y, crop_rate):

    try:
        video = cv2.VideoCapture(filename)

    except:
        print('cannot open video')
        exit(0)

    # get video rate and change variable unit from time to frame num
    fps = int(video.get(cv2.CAP_PROP_FPS))
    image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]
    # adjust video rate to keep genuine broadcast rate

    # calc image-crop-region crop -> expaned to original frame geometry
    offset_yt = image_res[0] * ((1.0 - crop_rate) * 0.5 + image_offset_y)
    offset_xl = image_res[1] * (1.0 - crop_rate) * 0.5
    crop_value = [int(offset_yt),
                  int(offset_yt + image_res[0] * crop_rate),
                  int(offset_xl),
                  int(offset_xl + image_res[1] * crop_rate)
                  ]

    return video, image_res, fps, crop_value


def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()


def getAtrrib(root, id):
    for pedestrian in root.iter("pedestrian"):
        if pedestrian.attrib.get("id") == id:
            return pedestrian

    return None


def cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate):
    # frame_list = []
    image_res = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))]
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    writer = cv2.VideoWriter(video_name, fourcc, frame_rate, (image_res[1], image_res[0]))
    for index in range(start_frame, end_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame[crop_value[0]:crop_value[1], crop_value[2]:crop_value[3]], dsize=None, fx=expand_rate, fy=expand_rate)
            writer.write(frame)
            # frame_list.append(frame)
        else:
            print("failed to get frame")
            break

    writer.release()


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


def getAnchor(track, start_frame, end_frame, crop_value, crop_rate):
    anchor_list = []
    for frame in range(start_frame, end_frame+1):
        is_found = False
        for box in track.iter("box"):
            if int(box.get("frame")) == frame:
                anchor = {
                    # int((float(box.get('xbr')) - crop_value[2]) * (1 / crop_rate)),
                    # int((float(box.get('xtl')) - crop_value[2]) * (1 / crop_rate)),
                    # int((float(box.get('ybr')) - crop_value[0]) * (1 / crop_rate)),
                    # int((float(box.get('ytl')) - crop_value[0]) * (1 / crop_rate)),
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
    image_offset_y = 0.2
    crop_rate =  0.6
    expand_rate = 1.0 / crop_rate
    max_after_length = 2*30 # second x frame_rate
    int_length_list = [1,0, 3.0, 5.0, 7.0 ,9.0, 12.0]
    annt_attribute_root = getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_name))
    annt_root = getXmlRoot("{}/annotations/{}_annt.xml".format(base_dir, video_name))
    ego_vehicle_root = getXmlRoot("{}/annotations_vehicle/{}_obd.xml".format(base_dir, video_name))
    video_file = "{}/PIE_clips/{}.mp4".format(base_dir, video_name)
    video, _, _, crop_value = getVideo(video_file, image_offset_y, crop_rate)

    for track in annt_root.iter("track"):
        if track.get("label") == "pedestrian":
            for box_attrib in track[0].iter("attribute"):
                if box_attrib.get("name") == "id":
                    ped_id = box_attrib.text
                    print(ped_id)
            ped_attrib = getAtrrib(annt_attribute_root, ped_id)
            max_len = (int(ped_attrib.get("critical_point")) - int(track[0].get("frame")))/30
            # short video
            if int_length_list[0] > max_len:
                name = "{}_{}".format(ped_id, 0.0)
                start_frame = int(track[0].get("frame"))
                end_frame = int(track[-1].get("frame"))
                if result_dict.get(ped_id).get(start_frame) is None:
                    prediction = result_dict.get(ped_id).get(min(result_dict.get(ped_id).keys()))
                else:
                    prediction = result_dict.get(ped_id).get(start_frame)

                print("short", name, start_frame, int(ped_attrib.get("crossing_point")))
                # cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
                # print("short video : {} - {}s {}->{}".format(ped_id, max_len, start_frame, end_frame))

                video_database = {
                    "video_file" : video_file,
                    "id" : ped_id,
                    "label" : "pedestrian",
                    "int_length" : 0,
                    "prob" : float(ped_attrib.get("intention_prob")),
                    # "results" : float(ped_attrib.get("intention_prob")),
                    "results" : prediction,
                    "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                    "future_direction" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                    "critical_point" : int(ped_attrib.get("critical_point")),
                    "crossing_point" : min(int(ped_attrib.get("crossing_point")), int(ped_attrib.get("critical_point"))+max_after_length),
                    "start_frame" : start_frame
                }

                database[name] = video_database

            # cut video each size
            for int_length in int_length_list:
                if int_length > max_len:
                    break
                name = "{}_{}".format(ped_id, int_length)
                # video_name = "{}/extracted_data/{}_ped_{}_{}.mp4".format(base_dir, video_list[0], ped_id, int_length)
                start_frame = int(int(ped_attrib.get("critical_point")) - int_length * 30)
                end_frame = int(track[-1].get("frame"))
                if result_dict.get(ped_id).get(start_frame) is None:
                    prediction = result_dict.get(ped_id).get(min(result_dict.get(ped_id).keys()))
                else:
                    prediction = result_dict.get(ped_id).get(start_frame)
                print("long", name, start_frame, int(ped_attrib.get("crossing_point")))
                # cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
                # print("start cut :{}- {}s {}->{}".format(ped_id, int_length, start_frame, end_frame))

                video_database = {
                    "video_file" : video_file,
                    "id" : ped_id,
                    "label" : "pedestrian",
                    "int_length" : int_length,
                    "prob" : float(ped_attrib.get("intention_prob")),
                    # "results" : float(ped_attrib.get("intention_prob")),
                    "results" : prediction,
                    "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                    "future_direction" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                    "critical_point" : float(ped_attrib.get("critical_point")),
                    "crossing_point" : min(int(ped_attrib.get("crossing_point")), int(ped_attrib.get("critical_point"))+max_after_length),
                    "start_frame" : start_frame
                }

                database[name] = video_database


        elif track.get("label") == "traffic_light":
            for box_attrib in track[0].iter("attribute"):
                if box_attrib.get("name") == "id":
                    tl_id = box_attrib.text
                    print(tl_id)
            max_len = (int(track[-1].get("frame")) - int(track[0].get("frame")))/30

            # short video
            if int_length_list[0] > max_len:
                name = "{}_{}".format(tl_id, 0.0)
                # video_name = "{}/extracted_data/{}_tl_{}_0.mp4".format(base_dir, video_list[0], tl_id)
                start_frame = int(track[0].get("frame"))
                end_frame = int(track[-1].get("frame"))
                print(name, start_frame, end_frame)
                # cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
                # print("short video : {} - {}s {}->{}".format(tl_id, max_len, start_frame, end_frame))

                video_database = {
                    "video_file" : video_file,
                    "id" : tl_id,
                    "label" : "traffic_light",
                    "int_length" : 0,
                    "prob" : random.random(),
                    "results" : random.random(),
                    "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                    "future_direction" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                    "critical_point" : float(track[-1].get('frame')),
                    "crossing_point" : int(track[-1].get('frame')),
                    "start_frame" : start_frame
                }

                database[name] = video_database

            # cut video each size
            for int_length in int_length_list:
                if int_length > max_len:
                    break

                name = "{}_{}".format(tl_id, int_length)
                # video_name = "{}/extracted_data/{}_tl_{}_{}.mp4".format(base_dir, video_list[0], tl_id, int_length)
                start_frame = int(int(track[-1].get("frame")) - int_length * 30)
                end_frame = int(track[-1].get("frame"))
                print(name, start_frame, end_frame)
                # cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
                # print("start cut :{}- {}s {}->{}".format(tl_id, int_length, start_frame, end_frame))

                video_database = {
                    "video_file" : video_file,
                    "id" : tl_id,
                    "label" : "traffic_light",
                    "int_length" : int_length,
                    "prob" : random.random(),
                    "results" : random.random(),
                    "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                    "future_direction" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                    "critical_point" : int(track[-1].get('frame')),
                    "crossing_point" : int(track[-1].get('frame')),
                    "start_frame" : start_frame
                }

                database[name] = video_database


base_dir = "/media/kuriatsu/SamsungKURI/PIE_data"

result_file_list = [
    base_dir + "/extracted_data/predict/test/result_0-150.pkl",
    base_dir + "/extracted_data/predict/test/result_150-300.pkl",
    base_dir + "/extracted_data/predict/test/result_300-450.pkl",
    base_dir + "/extracted_data/predict/test/result_450-600.pkl",
    base_dir + "/extracted_data/predict/test/result_600-719.pkl",
    # base_dir + "/extracted_data/predict/val/result_0-150.pkl",
    # base_dir + "/extracted_data/predict/val/result_151-243.pkl",
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

    with open("{}/extracted_data/database_result_valid.pkl".format(base_dir), "wb") as f:
        pickle.dump(dict(database), f)



## single process
# database = {}
# for video in video_list:
#     process(database, video)
#
# with open("{}/extracted_data/database.pkl".format(base_dir), "wb") as f:
#     pickle.dump(database, f)
