#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import glob
import sys
import pickle
from multiprocessing import Pool, Manager

def getXmlRoot(filename):

    # try:
    tree = ET.parse(filename)
    return tree.getroot()


def getVideo(filename):

    try:
        video = cv2.VideoCapture(filename)

    except:
        print('cannot open video')
        exit(0)

    return video


def main(annotation_file, base_dir, set, out_data):

    label = {"red":[1,0,0], "yellow":[0,1,0], "green":[0,0,1], "__undefined__":[0,0,0]}

    # get annotation
    annt_root = getXmlRoot(annotation_file)

    # get video
    clip_name = annotation_file.split("/")[-1].replace("_annt.xml", "")
    video_file = f"{base_dir}/PIE_clips/{set}/{clip_name}.mp4"
    video = getVideo(video_file)

    for track in annt_root.iter("track"):
        if track.get("label") != "traffic_light":
            continue
        for box in track.iter("box"):
            print(box.get("frame"))
            video.set(cv2.CAP_PROP_POS_FRAMES, float(box.get("frame")))
            ret, frame = video.read()

            if not ret:
                print("failed to get frame")
                continue

            # crop tr
            tl_bb = frame[
                int(float(box.get("ytl"))) : int(float(box.get("ybr"))),
                int(float(box.get("xtl"))) : int(float(box.get("xbr")))
                ]
            std_img = cv2.resize(tl_bb.astype("uint8"), dsize=(32, 32))

            # get annotation
            state = None
            id = None
            type = None
            for attrib in box.iter("attribute"):
                if attrib.get("name") == "type":
                    type = attrib.text
                elif attrib.get("name") == "state":
                    state = label.get(attrib.text)
                elif attrib.get("name") == "id":
                    id = attrib.text

            if type != "regular":
                continue

            buf_data = {
                "image" : std_img,
                "state" : state,
                "id"    : id,
                "set"   : set,
                "video" : clip_name,
            }
            out_data.append(buf_data)
            # cv2.imshow("tl", std_img)
            # cv2.waitKey(1)
            # print(state)

if __name__ == "__main__":
    base_dir = "/media/kuriatsu/InternalHDD/PIE"
    # out_data = []
    with Manager() as manager:
        p = Pool(8)
        out_data = manager.list()
        for set in ["set01", "set02", "set03", "set04", "set05", "set06"]:
            for annotation_file in glob.iglob(base_dir+"/annotations/"+set+"/*.xml"):
                p.apply_async(main, args=(annotation_file, base_dir, set, out_data))

        p.close()
        p.join()

        with open("/media/kuriatsu/InternalHDD/PIE/tlr/database.pickle", "wb") as f:
            pickle.dump(list(out_data), f)
