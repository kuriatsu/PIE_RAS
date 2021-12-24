#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import threading
import random
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
    for i in range(start_frame, len(vehicle_root)):
        dist_buf += float(vehicle_root[i].get('OBD_speed')) * 0.03 / 3.6
        future_angle = float(vehicle_root[end_frame].get('yaw')) - float(vehicle_root[i].get('yaw'))
        if dist_buf > 50:
            break

    if 1.0 < abs(future_angle) % 3.14 < 2.0:
        if future_angle > 0.0:
            return 'right'
        else:
            return 'left'
    else:
        return 'straight'


def getAnchor(track, start_frame, end_frame, crop_value, crop_rate):
    anchor_list = []
    for frame in range(start_frame, end_frame+1):
        for box in track.iter("box"):
            if int(box.get("frame")) == frame:
                anchor = [
                    int((float(box.get('xbr')) - crop_value[2]) * (1 / crop_rate)),
                    int((float(box.get('xtl')) - crop_value[2]) * (1 / crop_rate)),
                    int((float(box.get('ybr')) - crop_value[0]) * (1 / crop_rate)),
                    int((float(box.get('ytl')) - crop_value[0]) * (1 / crop_rate)),
                    ]
                break
        anchor_list.append(anchor)

    return anchor_list


result_file_list = [
    "/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_0-150.pkl",
    "/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_151-300.pkl",
    "/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_301-450.pkl",
    "/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_451-600.pkl",
    "/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/predict/test/result_601-719.pkl",
    ]

# load prediction result
prediction_data = []
for file in result_file_list:
    with open(file, "rb") as f:
        prediction_data+=pickle.load(f)

# get final prediction result for each pedestrian
result_dict = {}
for buf in prediction_data:
    result_dict[buf.get("ped_id")] = float(buf.get("res"))

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
]


base_dir = "/run/media/kuriatsu/SamsungKURI/PIE_data"
annt_attribute_root = getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_list[0]))
annt_root = getXmlRoot("{}/annotations/{}_annt.xml".format(base_dir, video_list[0]))
ego_vehicle_root = getXmlRoot("{}/annotations_vehicle/{}_obd.xml".format(base_dir, video_list[0]))
video_file = "{}/PIE_clips/{}.mp4".format(base_dir, video_list[0])
image_offset_y = 0.2
crop_rate =  0.6
video, _, _, crop_value = getVideo(video_file, image_offset_y, crop_rate)
expand_rate = 1.0 / crop_rate

# {"video_path" :
#     {"id" : str,
#      "label" : str,
#      "length" : float,
#      "prob" : float,
#      "results" : float,
#      "anchor" : [[xbr, xtl, ybr, ytl],...],
#      "future_traj" : str,
#      "critical_point" : float,
#      "crossing_point" : int,
#     }
# }
database = {}
int_length_list = [1.0, 3.0, 5.0, 7.0 ,9.0]

for track in annt_root.iter("track"):
    if track.get("label") == "pedestrian":
        for box_attrib in track[0].iter("attribute"):
            if box_attrib.get("name") == "id":
                ped_id = box_attrib.text
        ped_attrib = getAtrrib(annt_attribute_root, ped_id)
        max_len = (int(ped_attrib.get("critical_point")) - int(track[0].get("frame")))/30

        # short video
        if int_length_list[0] > max_len:
            video_name = "{}/extracted_data/{}_ped_{}_0.mp4".format(base_dir, video_list[0], ped_id)
            start_frame = int(track[0].get("frame"))
            end_frame = int(track[-1].get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
            print("short video : {} - {}s {}->{}".format(ped_id, max_len, start_frame, end_frame))

            video_database = {
                "id" : ped_id,
                "label" : "pedestrian",
                "int_length" : int_length,
                "prob" : ped_attrib.get("intention_prob"),
                "results" : result_dict.get(ped_id),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : float(ped_attrib.get("critical_point")) - start_frame,
                "crossing_point" : int(ped_attrib.get("crossing_point")) - start_frame,
            }

            database[video_name] = video_database

        # cut video each size
        for int_length in int_length_list:
            if int_length > max_len:
                break
            video_name = "{}/extracted_data/{}_ped_{}_{}.mp4".format(base_dir, video_list[0], ped_id, int_length)
            start_frame = int(int(ped_attrib.get("critical_point")) - int_length * 30)
            end_frame = int(track[-1].get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
            print("start cut :{}- {}s {}->{}".format(ped_id, int_length, start_frame, end_frame))

            video_database = {
                "id" : ped_id,
                "int_length" : int_length,
                "prob" : ped_attrib.get("intention_prob"),
                "results" : result_dict.get(ped_id),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : float(ped_attrib.get("critical_point")) - start_frame,
                "crossing_point" : int(ped_attrib.get("crossing_point")) - start_frame,
            }

            database[video_name] = video_database


    if track.get("label") == "traffic_light":
        for box_attrib in track[0].iter("attribute"):
            if box_attrib.get("name") == "id":
                tl_id = box_attrib.text
        max_len = (int(track[-1].get("frame")) - int(track[0].get("frame")))/30

        # short video
        if int_length_list[0] > max_len:
            video_name = "{}/extracted_data/{}_tl_{}_0.mp4".format(base_dir, video_list[0], tl_id)
            start_frame = int(track[0].get("frame"))
            end_frame = int(track[-1].get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
            print("short video : {} - {}s {}->{}".format(tl_id, max_len, start_frame, end_frame))

            video_database = {
                "id" : tl_id,
                "label" : "traffic_light",
                "int_length" : int_length,
                "prob" : random.random(),
                "results" : random.random(),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : float(track[-1].get('frame')) - start_frame,
                "crossing_point" : int(track[-1].get('frame')) - start_frame,
            }

            database[video_name] = video_database

        # cut video each size
        for int_length in int_length_list:
            if int_length > max_len:
                break

            video_name = "{}/extracted_data/{}_tl_{}_{}.mp4".format(base_dir, video_list[0], tl_id, int_length)
            start_frame = int(int(track[-1].get("frame")) - int_length * 30)
            end_frame = int(track[-1].get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, crop_value, expand_rate)
            print("start cut :{}- {}s {}->{}".format(tl_id, int_length, start_frame, end_frame))

            video_database = {
                "id" : tl_id,
                "int_length" : int_length,
                "prob" : random.random(),
                "results" : random.random(),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : float(track[-1].get('frame')) - start_frame,
                "crossing_point" : int(track[-1].get('frame')) - start_frame,
            }

            database[video_name] = video_database

with open("{}/extracted_data/database.pkl".format(base_dir), "wb") as f:
    pickle.dump(database, f)
