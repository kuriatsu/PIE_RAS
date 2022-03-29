#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import glob
import os
import math
import xml.etree.ElementTree as ET
import cv2

import matplotlib.pyplot as plt
import seaborn as sns


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

def getAtrrib(root, tag, attrib, attrib_target):
    for i in root.iter(tag):
        if i.attrib.get(attrib) == attrib_target:
            return i.text

    return None

base_dir = "/media/kuriatsu/SamsungKURI/PIE_data"
video_list = []
for i in range(1, 20):
    video_name = "set03/video_{}".format(str(i).zfill(4))
    annt_attribute_root = getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_name))
    annt_root = getXmlRoot("{}/annotations/{}_annt.xml".format(base_dir, video_name))
    ego_vehicle_root = getXmlRoot("{}/annotations_vehicle/{}_obd.xml".format(base_dir, video_name))
    video_file = "{}/PIE_clips/{}.mp4".format(base_dir, video_name)
    print(video_file)
    video, image_res, fps = getVideo(video_file)

    for track in annt_root.iter("track"):
        if track.get("label") == "traffic_light":
            tl_id = getAtrrib(track[-1], "attribute", "name", "id")
            video.set(cv2.CAP_PROP_POS_FRAMES, int(track[-10].get("frame")))
            ret, frame = video.read()
            if ret:
                cv2.rectangle(
                    frame,
                    (int(float(track[-10].get('xtl'))), int(float(track[-10].get('ytl')))),
                    (int(float(track[-10].get('xbr'))), int(float(track[-10].get('ybr')))),
                    (0, 255, 0),
                    2
                    )
                cv2.imwrite("{}/images/tl/{}.jpg".format(base_dir, tl_id), frame)
