#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import glob


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

    return videl


def main(set, out_data):

    label = {"red":[1,0,0], "yellow":[0,1,0], "green":[0,0,1], "__undefined__":[0,0,0]}
    base_dir = "/media/kuriatsu/InternalHDD/PIE"

    for annotation_file in glob.iglob(base_dir+"/annotations/"+set+"/*.xml"):
        # get annotation
        root = getXmlRoot(annotation_file)

        # get video
        clip_name = annotation_file.split("/")[-1].replace("_annt.xml", "")
        video_file = base_dir + "/PIE_clips/" + clip_name + ".mp4"
        video = getVideo(video_file)

        for track in annt_root.iter("track"):
            if track.get("label") != "traffic_light":
                continue
            for box in track.iter("box"):
                video.set(cv2.CAP_PROP_POS_FRAMES, box.get("frame"))
                ret, frame = video.read()

                if not ret:
                    print("failed to get frame")
                    continue

                # crop tr
                tl_bb = frame[
                    float(box.get("ytl")) : float(box.get("ybr")),
                    float(box.get("xtl")) : float(box.get("xbr"))
                    ]
                std_img = cv2.resize(image.astype("uint8"), dsize=(32, 32))

                # get annotation
                state = None
                id = None
                for attrib in box.iter("attribute"):
                    if attrib.get("name") == "type" and attrib.text != "regular":
                        state = None
                        break
                    elif attrib.get("name") == "state":
                        state = label.get(attrib.get("state"))
                    elif attrib.get("name") == "id":
                        id = attrib.text

                buf_data = {
                    "image" : std_img,
                    "state" : state,
                    "id"    : id,
                }
                out_data.append(buf_data)

    return out_data

if __name__ == "__main__":
    set = sys.argv[1]
    out_data = []
    main(set, out_data)
