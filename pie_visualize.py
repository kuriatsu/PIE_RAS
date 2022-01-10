#! /usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import pickle
import glob
import numpy as np
import time
import random
import csv
import xml.etree.ElementTree as ET

class PIEVisualize():

    def __init__(self):
        self.frame_count = 0
        self.video_res = None
        self.video_fps = None
        cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)

    def __enter__(self):
        return self

    def getVideo(self, filename):
        try:
            video = cv2.VideoCapture(filename)

        except:
            print('cannot open video')
            exit(0)
        # get video rate and change variable unit from time to frame num
        self.video_fps = int(video.get(cv2.CAP_PROP_FPS))
        self.video_res = {"y" : video.get(cv2.CAP_PROP_FRAME_HEIGHT), "x" : video.get(cv2.CAP_PROP_FRAME_WIDTH)}

        return video

    def getXmlRoot(self, filename):

        # try:
        tree = ET.parse(filename)
        return tree.getroot()

    def getAnchor(self, annt, frame):
        anchor_list = []
        for box in annt.iter("box"):
            if int(box.get("frame")) == frame:
                for box_attrib in box.iter("attribute"):
                    if box_attrib.get("name") == "id":
                        ped_id = box_attrib.text

                anchor = {
                    "xbr": int(float(box.get('xbr'))),
                    "xtl": int(float(box.get('xtl'))),
                    "ybr": int(float(box.get('ybr'))),
                    "ytl": int(float(box.get('ytl'))),
                    }
                anchor_list.append({"id":ped_id, "anchor": anchor})

        return anchor_list

    def getAtrrib(self, root, id):
        for pedestrian in root.iter("pedestrian"):
            if pedestrian.get("id") == id:
                return pedestrian

        return None

    def renderInfo(self, frame, frame_count, annt_attribute, annt, ego_vehicle):
        """add information to the image

        """
        anchor_list = self.getAnchor(annt, frame_count)
        for obj_anchor in anchor_list:
            cv2.rectangle(
                frame,
                (obj_anchor.get("anchor").get('xtl'), obj_anchor.get("anchor").get('ytl')),
                (obj_anchor.get("anchor").get('xbr'), obj_anchor.get("anchor").get('ybr')),
                (0, 0, 255),
                2
                )
            cv2.putText(
                frame,
                'id:{}'.format(obj_anchor.get('id')),
                # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
                (obj_anchor.get("anchor").get('xtl'), obj_anchor.get("anchor").get('ytl') - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA
                )

            attrib = self.getAtrrib(annt_attribute, obj_anchor.get("id"))
            if attrib is not None:
                # show probability
                cv2.putText(
                    frame,
                    'prob:{}'.format(attrib.get('intention_prob')),
                    # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
                    (int(obj_anchor.get("anchor").get('xtl')), int(obj_anchor.get("anchor").get('ytl') - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA
                    )

        last_frame = min(len(ego_vehicle), frame_count+90)
        diff = float(ego_vehicle[last_frame].get("yaw")) - float(ego_vehicle[frame_count].get("yaw"))
        prod = np.sin(diff)
        cv2.putText(
            frame,
            'prod (3s-now){:.001f}'.format(prod),
            # 'angle {:.001f} 3s{:.001f}'.format(float(ego_vehicle[frame_count].get("yaw")), float(ego_vehicle[last_frame].get("yaw"))),
            # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
            (int(self.video_res.get("x")//2), int(self.video_res.get("y") - 60)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
            )
        cv2.putText(
            frame,
            'gps_speed:{:.01f}'.format(float(ego_vehicle[frame_count].get("GPS_speed"))),
            # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
            (int(self.video_res.get("x")//2), int(self.video_res.get("y") - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
            )
        cv2.putText(
            frame,
            'obd_speed:{:.01f}'.format(float(ego_vehicle[frame_count].get("OBD_speed"))),
            # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
            (int(self.video_res.get("x")//2), int(self.video_res.get("y") - 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
            )
    def play(self, base_dir, video_name):
        annt_root = self.getXmlRoot("{}/annotations/{}_annt.xml".format(base_dir, video_name))
        annt_attribute_root = self.getXmlRoot("{}/annotations_attributes/{}_attributes.xml".format(base_dir, video_name))
        ego_vehicle_root = self.getXmlRoot("{}/annotations_vehicle/{}_obd.xml".format(base_dir, video_name))
        video_file = "{}/PIE_clips/{}.mp4".format(base_dir, video_name)
        video = self.getVideo(video_file)

        ret, frame = video.read()
        frame_count = 0

        while ret:
            start = time.time()
            self.renderInfo(frame, frame_count, annt_attribute_root, annt_root, ego_vehicle_root) # add info to the frame
            cv2.moveWindow("video", 0, 0)
            cv2.imshow("video", frame) # render

            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (30) - (time.time() - start))), 1)
            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            ret, frame = video.read()
            frame_count += 1

        video.release()

    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # pie_visualize = PIEVisualize()
    with PIEVisualize() as pie_visualize:
        pie_visualize.play("/media/kuriatsu/SamsungKURI/PIE_data", "set03/video_0003")
