#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pickle

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


def cutVideo(video, start_frame, end_frame, video_name, res):
    # frame_list = []
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    writer = cv2.VideoWriter(video_name, (res[1], res[0]))
    for index in range(start_frame, end_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame[crop_value[0]:crop_value[1], crop_value[2]:crop_value[3]], dsize=None, fx=expand_rate, fy=expand_rate)
            writer.write(frame)
            # frame_list.append(frame)
        else:
            break

    writer.release()


def getVehicleDirection(vehicle_root, start_frame, end_frame):
    for i in range(start_frame, len(vehicle_root)):
        dist_buf += float(vehicle_root[i].attrib.get('OBD_speed')) * 0.03 / 3.6
        future_angle = float(vehicle_root[end_frame].attrib.get('yaw')) - float(vehicle_root[i].attrib.get('yaw'))
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
            if int(box.attrib.get("frame")) == frame:
                anchor = [
                    int((float(box.attrib.get('xbr')) - crop_value[2]) * (1 / crop_rate)),
                    int((float(box.attrib.get('xtl')) - crop_value[2]) * (1 / crop_rate)),
                    int((float(box.attrib.get('ybr')) - crop_value[0]) * (1 / crop_rate)),
                    int((float(box.attrib.get('ytl')) - crop_value[0]) * (1 / crop_rate)),
                    ]
                break
        anchor_list.append(anchor)

    return anchor_list


result_file_list = [
    "/home/kuriatsu/Documents/data/pie_predict/result_0-150.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_151-300.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_301-450.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_451-600.pkl",
    "/home/kuriatsu/Documents/data/pie_predict/result_601-719.pkl",
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

int_length_list = [1.0, 3.0, 5.0, 7.0 ,9.0]
image_offset_y = 0.2
crop_rate =  0.6
video, _, _, crop_value = PieLib.getVideo(video_file, image_offset_y, crop_rate)
image_res = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))]
frame_rate = int(video.get(cv2.CAP_PROP_FPS))
expand_rate = 1.0 / crop_rate

annt_attribute_root = getXmlRoot("/home/kuriatsu/media/SamsungKURI/PIE_data/annotations_attributes/{}/{}_attributes.xml".format(set_list[0], video_list[0]))
annt_root = getXmlRoot("/home/kuriatsu/Documents/data/annotations/{}/{}_annt.xml".format(set_list[0], video_list[0]))
ego_vehicle_root = getXmlRoot("/home/kuriatsu/media/SamsungKURI/PIE_data/annotations_attributes/{}/{}_vehicle.xml".format(set_list[0], video_list[0]))

# {"video_path" :
#     {"id" : str,
#      "label" : str,
#      "length" : float,
#      "prob" : float,
#      "results" : float,
#      "anchor" : [[xbr, xtl, ybr, ytl],...],
#      "future_traj" : str,
#      "critical_point" : int,
#      "crossing_point" : int,
#     }
# }
database = {}

for track in annt_root.iter("track"):
    if track.attrib.get("label") == "pedestrian":
        for box_attrib in track[0].iter("box"):
            if box_attrib.get("name") == "id":
                ped_id = box_attrib.text
        ped_attrib = getAtrrib(annt_attribute_root, ped_id).attrib.get("critical_point"))
        max_len = int(ped_attrib.attrib.get("critical_point")) - int(track[0].attrib.get("frame"))

        # short video
        if int_length_list[0] > max_len:
            video_name = "/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/{}_{}_ped_{}_0.mp4".format(set_list[0], video_list[0], ped_id)
            start_frame = int(track[0].attrib.get("frame"))
            end_frame = int(track[-1].attrib.get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, image_res)

            video_database = {
                "id" : ped_id,
                "label" : "pedestrian",
                "int_length" : int_length,
                "prob" : ped_attrib.attrib.get("intention_prob"),
                "results" : result_dict.get(ped_id),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : int(ped_attrib.attrib.get("intention_prob")) - start_frame,
                "crossing_point" : int(ped_attrib.attrib.get("crossing_point")) - start_frame,
            }

            database[video_name] = video_database

        # cut video each size
        for int_length in int_length_list:
            if int_length > max_len:
                break

            video_name = "/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/{}_{}_ped_{}_{}.mp4".format(set_list[0], video_list[0], ped_id, int_length)
            start_frame = int(ped_attrib.attrib.get("critical_point")) - int_length * 30
            end_frame = int(track[-1].attrib.get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, image_res)

            video_database = {
                "id" : ped_id,
                "int_length" : int_length,
                "prob" : ped_attrib.attrib.get("intention_prob"),
                "results" : result_dict.get(ped_id),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : int(ped_attrib.attrib.get("intention_prob")) - start_frame,
                "crossing_point" : int(ped_attrib.attrib.get("crossing_point")) - start_frame,
            }

            database[video_name] = video_database


    if track.attrib.get("label") == "traffic_light":
        for box_attrib in track[0].iter("box"):
            if box_attrib.get("name") == "id":
                tl_id = box_attrib.text
        max_len = int(track[-1].attrib.get("frame")) - int(track[0].attrib.get("frame"))

        # short video
        if int_length_list[0] > max_len:
            video_name = "/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/{}_{}_tl_{}_0.mp4".format(set_list[0], video_list[0], tl_id)
            start_frame = int(track[0].attrib.get("frame"))
            end_frame = int(track[-1].attrib.get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, image_res)

            video_database = {
                "id" : tl_id,
                "label" : "traffic_light",
                "int_length" : int_length,
                "prob" : random.random(),
                "results" : random.random(),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : int(track[-1].attrib.get('frame')) - start_frame,
                "crossing_point" : int(track[-1].attrib.get('frame'))) - start_frame,
            }

            database[video_name] = video_database

        # cut video each size
        for int_length in int_length_list:
            if int_length > max_len:
                break

            video_name = "/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/{}_{}_tl_{}_{}.mp4".format(set_list[0], video_list[0], tl_id, int_length)
            start_frame = int(track[0].attrib.get("frame")) - int_length * 30
            end_frame = int(track[-1].attrib.get("frame"))
            cutVideo(video, start_frame, end_frame, video_name, image_res)

            video_database = {
                "id" : tl_id,
                "int_length" : int_length,
                "prob" : random.random(),
                "results" : random.random(),
                "anchor" : getAnchor(track, start_frame, end_frame, crop_value, crop_rate),
                "future_traj" : getVehicleDirection(ego_vehicle_root, start_frame, end_frame),
                "critical_point" : int(track[-1].attrib.get('frame')) - start_frame,
                "crossing_point" : int(track[-1].attrib.get('frame'))) - start_frame,
            }

            database[video_name] = video_database
