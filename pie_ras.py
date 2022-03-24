#! /usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import pickle
import glob
import numpy as np
import time
import random
import csv
import copy
import pygame
from pygame.locals import JOYBUTTONUP, JOYBUTTONDOWN, JOYHATMOTION

class PIERas():
    def __init__(self, recognition_type):

        self.log_file = "./data/PIE_data/experiment/log_data.csv"
        self.recognition_type = recognition_type # "traj", "tl", "int"
        self.hmi_image_offset_rate_y = 0.2 # rate
        self.hmi_crop_margin = 40 # px
        self.hmi_crop_rate_max = 0.4
        self.hmi_crop_rate = 0.6
        self.hmi_res = {"x": 1920, "y": 1080}
        self.windshield_anchor = {
            "xtl": 0,
            "xbr": 1920,
            "ytl": int((1-1920/3840)*0.5*1080)+200,
            "ybr": int((0.5+1920/3840*0.5)*1080)+200,
            }
        self.windshield_res = {"x": 3840, "y": 1080}
        self.video_res = None
        self.video_fps = None

        self.frame_count = 0
        self.start_time = 0

        self.hmi_anchor = None
        self.target_anchor = None
        self.icon_dict = self.prepareIcon("./data/pictogram", recognition_type)
        self.icon_is_right = False

        self.log = []



        cv2.namedWindow("windshield", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("windshield", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("windshield", 0, 0)
        cv2.namedWindow("hmi", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("hmi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("hmi", self.windshield_res.get("x"), 0)

        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

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


    def cropResizeFrame(self, frame, hmi_anchor, video_res):
        """
        crop_size:[xl, xr, yl, yr]
        """
        expand_rate = video_res.get("x")/(hmi_anchor.get("xbr")-hmi_anchor.get("xtl"))
        return cv2.resize(frame[hmi_anchor.get("ytl"):hmi_anchor.get("ybr"), hmi_anchor.get("xtl"):hmi_anchor.get("xbr")], dsize=None, fx=expand_rate, fy=expand_rate)


    def prepareIcon(self, dirname, recognition_type):
        """
        file = dirname
        icon_dict = {
            "tf_green" : {'roi': roi, 'mask_inv': mask_inv, 'icon_fg': icon_fg}
            "tf_red"
            "walker_not_cross"
            "walker_cross_to_left"
            "walker_cross_to_right"
            "right_red"
            "left_red"
            "straight_red"
            "right_green"
            "left_green"
            "straight_green"
        }
        """
        icon_dict = {
            "path":{
                "red":{
                    "right" : self.getIcon(dirname+'/right_red.png'),
                    "left"  : self.getIcon(dirname+'/left_red.png'),
                    "straight"  : self.getIcon(dirname+'/straight_red.png'),
                    },
                "green":{
                    "right" : self.getIcon(dirname+'/right_green.png'),
                    "left"  : self.getIcon(dirname+'/left_green.png'),
                    "straight"  : self.getIcon(dirname+'/straight_green.png'),
                    },
            },
        }
        if recognition_type == "tl":
            icon_dict["recognition"] = {
                0: self.getIcon(dirname+'/tl_red.png'),
                1: self.getIcon(dirname+'/tl_green.png'),
                -1: self.getIcon(dirname+'/tl_init.png'),
            }
        elif recognition_type == "int":
            icon_dict["recognition"] = {
                "to_right":{
                    0: self.getIcon(dirname+'/int_cross_to_right.png'),
                    1: self.getIcon(dirname+'/int_no_cross_to_right.png'),
                    -1: self.getIcon(dirname+'/int_init_to_right.png'),
                },
                "to_left":{
                    0: self.getIcon(dirname+'/int_cross_to_left.png'),
                    1: self.getIcon(dirname+'/int_no_cross_to_left.png'),
                    -1: self.getIcon(dirname+'/int_init_to_left.png'),
                }
            }
        elif recognition_type == "traj":
            icon_dict["recognition"] = {
                "to_right":{
                    0: self.getIcon(dirname+'/traj_cross_to_right.png'),
                    1: self.getIcon(dirname+'/traj_no_cross.png'),
                    -1: self.getIcon(dirname+'/traj_init_to_right.png'),
                },
                "to_left":{
                    0: self.getIcon(dirname+'/traj_cross_to_left.png'),
                    1: self.getIcon(dirname+'/traj_no_cross.png'),
                    -1: self.getIcon(dirname+'/traj_init_to_left.png'),
                }
            }

        return icon_dict


    def getIcon(self, filename):
        img = cv2.imread(filename)
        name = filename.split('/')[-1].split('.')[-2]
        roi = img.shape[:2]
        img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        icon_fg = cv2.bitwise_and(img, img, mask=mask)
        return  {'roi': roi, 'mask_inv': mask_inv, 'icon_fg': icon_fg}


    def renderInfo(self, frame, database, obj_anchor, frame_count):
        """add information to the image

        """
        # icon name
        if database.get('label') == 'traffic_light':
            icon = self.icon_dict.get("recognition").get(self.target_state)
            # position of the icon
            icon_offset_y = 10.0
            print("acnho",self.recognition_type)
            icon_offset_x = int((icon.get("roi")[1] - (obj_anchor.get('xbr') - obj_anchor.get('xtl'))) * 0.5)
            icon_position = {
                'ytl': int(obj_anchor.get('ybr') + icon_offset_y),
                'xtl': int(obj_anchor.get('xtl') - icon_offset_x),
                'ybr': int(obj_anchor.get('ybr') + icon_offset_y + icon.get('roi')[0]),
                'xbr': int(obj_anchor.get('xtl') + icon.get('roi')[1] - icon_offset_x)
                }

        if database.get('label') == 'pedestrian':

            # if self.target_state == -1:
            if obj_anchor.get('xbr') < self.video_res.get("x") * 0.5:
                icon = self.icon_dict.get("recognition").get("to_right").get(self.target_state)
                self.icon_is_right = False
            else:
                icon = self.icon_dict.get("recognition").get("to_left").get(self.target_state)
                self.icon_is_right = True

            # if self.recognition_type == "int":
            # position of the icon
            icon_offset_y = 30.0
            icon_offset_x = int((icon.get("roi")[1] - (obj_anchor.get('xbr') - obj_anchor.get('xtl'))) * 0.5)
            icon_position = {
            'ytl': int(obj_anchor.get('ytl') - icon.get('roi')[0] - icon_offset_y),
            'xtl': int(obj_anchor.get('xtl') - icon_offset_x),
            'ybr': int(obj_anchor.get('ytl') - icon_offset_y),
            'xbr': int(obj_anchor.get('xtl') + icon.get('roi')[1] - icon_offset_x)
            }

            # elif self.recognition_type == "traj":
            #     # position of the icon
            #     icon_offset_y = int((icon.get("roi")[0] - (obj_anchor.get('ybr') - obj_anchor.get('ytl'))) * 0.5)
            #     icon_offset_x = 5
            #     if obj_anchor.get('xbr') < self.video_res.get("x") * 0.5:
            #         icon_position = {
            #         'ytl': int(obj_anchor.get('ytl') + icon_offset_y),
            #         'xtl': int(obj_anchor.get('xbr') + icon_offset_x),
            #         'ybr': int(obj_anchor.get('ytl') + icon_offset_y + icon.get('roi')[0]),
            #         'xbr': int(obj_anchor.get('xbr') + icon_offset_x + icon.get('roi')[1]),
            #         }
            #     else:
            #         icon_position = {
            #         'ytl': int(obj_anchor.get('ytl') + icon_offset_y),
            #         'xtl': int(obj_anchor.get('xtl') - icon_offset_x),
            #         'ybr': int(obj_anchor.get('ytl') + icon_offset_y + icon.get('roi')[0]),
            #         'xbr': int(obj_anchor.get('xtl') - icon_offset_x - icon.get('roi')[1]),
            #         }

        self.drawIcon(frame, icon, icon_position)
        cv2.rectangle(
            frame,
            (obj_anchor.get('xtl'), obj_anchor.get('ytl')),
            (obj_anchor.get('xbr'), obj_anchor.get('ybr')),
            (0, 255, 0) if self.target_state == 1 else (0, 0, 255),
            2
            )

        ## show probability
        # cv2.putText(
        #     image,
        #     '{:.01f}'.format(database['prob']),
        #     # '{:.01f}s'.format((database['critical_point'] - database['start_frame'] - frame_count) / self.fps),
        #     (self.current_database['xtl'], self.current_database['ytl'] + 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, color, 3, cv2.LINE_AA
        #     )

        # show progress
        rate = frame_count / (database.get("critical_point") - database.get("start_frame"))
        self.showProgress(frame, icon_position, rate)


    def calcAnchor(self, obj_anchor, hmi_anchor, video_res):
        expand_rate = video_res.get("x")/(hmi_anchor.get("xbr")-hmi_anchor.get("xtl"))
        out_anchor = {
            "xbr": int( (float(obj_anchor.get("xbr")) - hmi_anchor.get("xtl") ) * expand_rate),
            "xtl": int( (float(obj_anchor.get("xtl")) - hmi_anchor.get("xtl") ) * expand_rate),
            "ybr": int( (float(obj_anchor.get("ybr")) - hmi_anchor.get("ytl") ) * expand_rate),
            "ytl": int( (float(obj_anchor.get("ytl")) - hmi_anchor.get("ytl") ) * expand_rate),
            }
        return out_anchor

    def drawIcon(self, frame, icon_info, position):
        """draw icon to emphasize the target objects
        frame : frame
        position : PIE dataset info of the object in the frame
        """

        if position.get('ytl') < 0 or position.get('ybr') > self.video_res.get("y") or position.get('xtl') < 0 or position.get('xbr') > self.video_res.get("x"):
            print(f"icon is out of range y:{position.get('ytl')}-{position.get('ybr')}, x:{position.get('xtl')}-{position.get('xbr')}")
            return

        # put icon on frame
        try:
            roi = frame[position.get('ytl'):position.get('ybr'), position.get('xtl'):position.get('xbr')] # get roi from frame
            image_bg = cv2.bitwise_and(roi, roi, mask=icon_info['mask_inv']) # remove color from area for icon by filter
            buf = cv2.add(icon_info['icon_fg'], image_bg) # put icon of roi
            frame[position.get('ytl'):position.get('ybr'), position.get('xtl'):position.get('xbr')] = buf # replace frame region to roi

        except Exception as e:
            print(f'failed to put icon: {e}')


    def showProgress(self, frame, base_position, rate):

        color = (0, 255, 0) if self.target_state != -1 else (0, 0, 255)
        # straight probress bar
        cv2.line(
            frame,
            (base_position.get('xtl'), base_position.get('ybr') + 10),
            (base_position.get('xbr'), base_position.get('ybr') + 10),
            (0, 0, 0),
            thickness=8,
            lineType=cv2.LINE_AA
            )
        cv2.line(
            frame,
            (base_position.get('xtl'), base_position.get('ybr') + 10),
            (base_position.get('xtl') + int((base_position.get('xbr') - base_position.get('xtl')) * (1 - rate)), base_position.get('ybr') + 10),
            color,
            thickness=8,
            lineType=cv2.LINE_AA
            )


    def showTrajectory(self, frame, database):
        # future trajectory
        arrow_color = 'red' if self.target_state in [-1, 0] else 'green'
        arrow_info = self.icon_dict.get("path").get(arrow_color).get(database.get('future_direction'))
        arrow_position = {
            'ytl': int(self.hmi_res.get("y") - 250),
            'ybr': int(self.hmi_res.get("y") - 250 + arrow_info.get('roi')[0]),
            'xtl': int(self.hmi_res.get("x") / 2 - arrow_info.get('roi')[1]/2),
            'xbr': int(self.hmi_res.get("x") / 2 + arrow_info.get('roi')[1]/2)
            }
        self.drawIcon(frame, arrow_info, arrow_position)


    def getHMICropAnchor(self, obj_anchor, video_res):
        # obj_anchor = self.database.get("anchor")[frame_count-int(database.get("start_frame"))]
        # calc image-crop-region crop -> expaned to original frame geometry
        max_ytl = video_res.get("y") * ( (1.0 - self.hmi_crop_rate_max) * 0.5 + self.hmi_image_offset_rate_y )
        min_ybr = video_res.get("y") * (0.5 + self.hmi_crop_rate_max*0.5 + self.hmi_image_offset_rate_y)
        max_xtl = video_res.get("x") * ( (1.0 - self.hmi_crop_rate_max)*0.5 )
        min_xbr = video_res.get("x") * (0.5 + self.hmi_crop_rate_max*0.5)
        # print("ytl:{}, ybr:{}, xtl:{}, xbr:{}".format(max_ytl, m+ video_res.get("y") * self.hmi_image_offset_rate_yin_ybr, max_xtl, min_xbr))
        extend_size_x = max(max_xtl - obj_anchor.get("xtl") + self.hmi_crop_margin,  obj_anchor.get("xbr") + self.hmi_crop_margin - min_xbr, 0)
        extend_size_y = max(max_ytl - obj_anchor.get("ytl") + self.hmi_crop_margin,  obj_anchor.get("ybr") + self.hmi_crop_margin - min_ybr, 0)
        # print(extend_size, video_res.get("y") - min_ybr, max_ytl)
        # if extend_size < min(video_res.get("y") - min_ybr, max_ytl):
        if extend_size_x == 0 and extend_size_y == 0:
            out_anchor = {
                "xbr": int(min_xbr),
                "xtl": int(max_xtl),
                "ybr": int(min_ybr),
                "ytl": int(max_ytl),
            }
        elif extend_size_x > extend_size_y:
            out_anchor = {
                "xbr": int(min_xbr + extend_size_x),
                "xtl": int(max_xtl - extend_size_x),
                "ybr": int(min_ybr + video_res.get("y") * extend_size_x / video_res.get("x")),
                "ytl": int(max_ytl - video_res.get("y") * extend_size_x / video_res.get("x")),
            }
        else:
            out_anchor = {
                "xbr": int(min_xbr + video_res.get("x") * extend_size_y / video_res.get("y")),
                "xtl": int(max_xtl - video_res.get("x") * extend_size_y / video_res.get("y")),
                "ybr": int(min_ybr + extend_size_y),
                "ytl": int(max_ytl - extend_size_y),
            }


        if self.hmi_image_offset_rate_y > 0.0 and out_anchor.get("ybr") > video_res.get("y"):
            out_anchor["ytl"] -= int(out_anchor.get("ybr") - video_res.get("y"))
            out_anchor["ybr"] = int(video_res.get("y"))

        elif self.hmi_image_offset_rate_y < 0.0 and out_anchor.get("ytl") < 0:
            out_anchor["ybr"] -= int(out_anchor.get("ytl"))
            out_anchor["ytl"] = 0

        if out_anchor.get("xtl") < 0 or out_anchor.get("ytl") < 0 or out_anchor.get("xbr") > video_res.get("x") or out_anchor.get("ybr") > video_res.get("y"):
            out_anchor = {
                "xbr": int(video_res.get("x")),
                "xtl": 0,
                "ybr": int(video_res.get("y")),
                "ytl": 0,
            }
            print("out of crop size")
            # print(out_anchor.get("xbr") - out_anchor.get("xtl"), out_anchor.get("ybr") - out_anchor.get("ytl"))

        return out_anchor


    def convertCvImageToPygame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        # return pygame.surfarray.make_surface(rgb_image)
        # 同じ画像サイズで生成済のSurfaceをキャッシュから取得する
        cache_key = rgb_image.shape
        cached_surface = self.pygame_surface_cache.get(cache_key)

        if cached_surface is None:
            # OpenCVの画像を元に、Pygameで画像を描画するためのSurfaceを生成する
            cached_surface = pygame.surfarray.make_surface(rgb_image)
            # Surfaceをキャッシュに追加
            self.pygame_surface_cache[cache_key] = cached_surface
        else:
            # 同じ画像サイズのSurfaceが見つかった場合は、すでに生成したSurfaceを使い回す
            pygame.surfarray.blit_array(cached_surface, rgb_image)

        return cached_surface


    def buttonCallback(self, key):
        """callback of enter key push, target is focused object
        """
        if self.target_anchor is None:
            return

        if key.value[0] == 0 and key.value[1] == 0:
            return
        # print(key)
        if self.recognition_type == "tl" and key.value[0] == 0:
            if key.value[1] == -1:
                self.target_state = 1
            elif key.value[1] == 1:
                self.target_state = 0
            self.saveLog("button")

        elif self.recognition_type in ["int", "traj"] and key.value[0] == 0:
            # if key.value[0] == 1:
            #     self.target_state = 1 if self.icon_is_right else 0
            # elif key.value[0] == -1:
            #     self.target_state = 0 if self.icon_is_right else 1
            if key.value[1] == -1:
                self.target_state = 0
            elif key.value[1] == 1:
                self.target_state = 1
            self.saveLog("button")


    def saveLog(self, int_method):
        if self.log[-1][5] is None:
            self.log[-1][2] += 1
            self.log[-1][3] = self.frame_count
            self.log[-1][4] = time.time() - self.start_time
            self.log[-1][5] = self.frame_count
            self.log[-1][6] = time.time() - self.start_time
            self.log[-1][7] = self.target_state
        else:
            self.log[-1][2] += 1
            self.log[-1][5] = self.frame_count
            self.log[-1][6] = time.time() - self.start_time
            self.log[-1][7] = self.target_state
        print("saved")

    def play(self, database, intention_value):

        # init variables
        self.frame_count = 0
        self.start_time = time.time()
        self.target_state = -1
        print("id:{}, prediction:{}, GT:{}, cross:{}".format(database.get("id"), database.get("results"), database.get("likelihood"), database.get("state")))
        # init log
        self.log.append([
            database.get("id"), # id
            database.get("int_length"),
            0, # int_count
            None, # first_int_frame
            None, # first_int_time
            None, # last_int_frame
            None, # last_int_time
            self.target_state, # last_state cross/green:1, stop/red:0, None:-1
            ])

        print(database.get("video_file"))
        video = self.getVideo(database.get("video_file"))
        video.set(cv2.CAP_PROP_POS_FRAMES, database.get("start_frame"))
        ret, frame = video.read()

        while ret and database.get("end_frame") - database.get("start_frame") > self.frame_count:
            start = time.time()
            if any(database.get("anchor")[self.frame_count]):
                hmi_crop_anchor = self.getHMICropAnchor(database.get("anchor")[self.frame_count], self.video_res)
                target_anchor = self.calcAnchor(database.get("anchor")[self.frame_count], hmi_crop_anchor, self.video_res)
                croped_frame = self.cropResizeFrame(frame, hmi_crop_anchor, self.video_res)
                if self.frame_count < (database.get("critical_point") - database.get("start_frame")):
                    self.renderInfo(croped_frame, database, target_anchor, self.frame_count) # add info to the frame
                    self.target_anchor = target_anchor
                else:
                    self.target_anchor = None

                self.showTrajectory(croped_frame, database)
                cv2.imshow("hmi", croped_frame) # render
            else:
                # show image without boundingbox
                croped_frame = copy.deepcopy(frame)
                self.showTrajectory(croped_frame, database)
                cv2.imshow("hmi", croped_frame) # render

            croped_frame = self.cropResizeFrame(frame, self.windshield_anchor, self.windshield_res)
            cv2.imshow("windshield", croped_frame) # render

            for e in pygame.event.get():
                if e.type == JOYHATMOTION:
                    self.buttonCallback(e)
                # elif e.type == KEYUP:
                #     if not self.is_pushed:
                #         self.buttonCallback(e.button)
                #     self.is_pushed = True
                # elif e.type == KEYUP:
                #     self.is_pushed = False

            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (30+9) - (time.time() - start))), 1)
            cv2.waitKey(sleep_time)

            self.frame_count += 1
            ret, frame = video.read()

        video.release()


    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))

        cv2.destroyAllWindows()

        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['id', "int_length", "int_count", "first_int_frame", "first_int_time", "last_int_frame", "last_int_time", "last_state"])
            writer.writerows(self.log)


if __name__ == "__main__":

    # with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_test.pkl", 'rb') as f:
    with open("./data/PIE_data/experiment/database.pkl", 'rb') as f:
        database = pickle.load(f)
    ids = random.choices(list(database.keys()), k=100)
    # ids = ["3_9_582_12.0"]
    # print(ids)

    # with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist/mistake_playlilst.csv", "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         ids = row

    with PIERas() as pie_ras:
        for id in ids:
            print(id.rsplit("_", 1)[0])
            if not id.rsplit("_", 1)[0].endswith("tl"):
                continue
            pie_ras.play(database.get(id), "result")
