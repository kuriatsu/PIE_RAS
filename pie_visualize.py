#! /usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import pickle
import glob
import numpy as np
import time

class PIEVisualize():
    def __init__(self):
        self.database = None

        self.hmi_image_offset_rate_y = 0.2
        self.hmi_crop_rate_max = 0.6
        self.hmi_crop_rate = 0.6
        self.hmi_res = [1920, 1080]
        self.hmi_anchor = None
        self.touch_area_rate = 2.0

        self.windshield_anchor = None
        self.windshield_res = [3840, 1080]

        self.video_res = None
        self.video_fps = None

        self.icon_dict = {}
        self.is_checked = False

        self.prepareEventHandler()
        self.prepareIcon("/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/pie_icons")
        cv2.namedWindow("hmi", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("hmi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.namedWindow("windshield", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("windshield", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    def getVideo(self, filename):
        try:
            video = cv2.VideoCapture(filename)

        except:
            print('cannot open video')
            exit(0)
        # get video rate and change variable unit from time to frame num
        self.video_fps = int(video.get(cv2.CAP_PROP_FPS))
        self.video_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]

        # self.windshield_anchor = {
        #     "xtl" : int(anchor_xtl),
        #     "xbr" : int(anchor_xtl + self.video_res[1]),
        #     "ytl" : int(anchor_ytl),
        #     "ybr" : int(anchor_ytl + self.video_res[0]),
        #     }
        return video


    def cropFrame(self, frame, crop_value, expand_rate):
        """
        crop_size:[xl, xr, yl, yr]
        """
        anchor = {
            "xtl" : int(self.video_res[1] * self.hmi_crop_rate),
            "xbr" : int(self.video_res[1] * (0.5 + self.hmi_crop_rate*0.5)),
            "ytl" : int(self.video_res[0] * (self.hmi_crop_rate + self.hmi_image_offset_rate_y)),
            "ybr" : int(self.video_res[0] * (0.5 + self.hmi_crop_rate*0.5 + self.hmi_image_offset_rate_y),
            }
        # frame_list = []

    def getHMICropRate(self):
        obj_anchor = self.calcAnchor(self.database.get("anchor")[frame_count-int(database.get("start_frame"))])
        # calc image-crop-region crop -> expaned to original frame geometry
        max_ytl = self.video_res[0] * ((1.0 - self.max_hmi_crop_rate)*0.5 + self.hmi_image_offset_rate_y)
        min_ybr = self.video_res[0] * (0.5 + self.max_hmi_crop_rate*0.5 - self.hmi_image_offset_rate_y)
        max_xtl = self.video_res[1] * ((1.0 - self.max_hmi_crop_rate)*0.5)
        min_xbr = self.video_res[1] * (0.5 + self.max_hmi_crop_rate*0.5)

        extend_size = max(
            max_xtl - obj_anchor.get()"xtl"),
            obj_anchor.get["xbr"] - min_xbr,
            max_ytl - obj_anchor.get("ytl"),
            obj_anchor.get["ybr"] - min_ybr,
            0,
        )

        extend_size = min(
            max_ytl,
            (self.video_res[0] - min_ybr)),
            extend_size,
        )

        self.hmi_crop_rate = self.hmi_crop_rate_max + self.video_res[0] * extend_rate


    def prepareIcon(self, filename):
        """
        file = filename
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
        for icon_file in glob.iglob(filename+'/*.png'):
            img = cv2.imread(icon_file)
            name = icon_file.split('/')[-1].split('.')[-2]
            roi = img.shape[:2]
            img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
            # ret, mask = cv2.threshold(img2grey, 200, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            icon_fg = cv2.bitwise_and(img, img, mask=mask)

            self.icon_dict[name] = {'roi': roi, 'mask_inv': mask_inv, 'icon_fg': icon_fg}


    def prepareEventHandler(self):
        """add mouse click callback to the window
        """
        cv2.namedWindow("hmi")
        cv2.setMouseCallback("hmi", self.touchCallback)


    def touchCallback(self, event, x, y, flags, param):
        """if mouce clicked, check position and judge weather the position is on the rectange or not
        """
        anchor = self.database.get("anchor")[self.frame_count-self.database.get("start_frame")]
        touch_area = {
            "xtl" : anchor.get("xtl") - self.touch_area_rate * (anchor.get("xtl") - anchor.get("xbr")),
            "xbr" : anchor.get("xbr") + self.touch_area_rate * (anchor.get("xtl") - anchor.get("xbr")),
            "ytl" : anchor.get("ytl") - self.touch_area_rate * (anchor.get("ytl") - anchor.get("ybr")),
            "ybr" : anchor.get("ybr") + self.touch_area_rate * (anchor.get("ytl") - anchor.get("ybr")),
        }
        # if the event handler is leftButtonDown
        if event == cv2.EVENT_LBUTTONDOWN and \
           touch_area['xtl'] < x < touch_area['xbr'] and \
           touch_area['ytl'] < y < touch_area['ybr']:

            self.log[-1] += [time.time(), 'touched', None]
            self.is_checked = not self.is_checked


    def buttonCallback(self, key):
        """callback of enter key push, target is focused object
        """
        self.log[-1] += [time.time(), 'pushed', key]
        self.is_checked = not self.is_checked


    def renderInfo(self, frame, database, frame_count):
        """add information to the image

        """
        anchor = self.calcAnchor(self.database.get("anchor")[frame_count-int(database.get("start_frame"))])
        expand_px = max(
            self.hmi_base_anchor.get("xtl") - anchor.get("xtl"),
            anchor.get("xbr") - self.hmi_base_anchor.get("xbr"),
            self.hmi_base_anchor.get("ytl") - anchor.get("ytl"),
            anchor.get("ybr") - self.hmi_base_anchor.get("ybr"),
            0,
        )
        expand_px = min(expand_px, self.video_res[0] - self.hmi_base_anchor)
        # icon name
        if database.get('label') == 'traffic_light':
            icon = self.icon_dict.get("tf_green") if self.is_checked else self.icon_dict.get("tf_red")

        if database.get('label') == 'pedestrian':
            if self.is_checked:
                icon = self.icon_dict.get("walker_cross")
            else:
                if anchor.get('xbr') < self.video_res[1] * 0.5:
                    icon = self.icon_dict.get("walker_cross_to_right")
                else:
                    icon = self.icon_dict.get("walker_cross_to_left")

        # position of the icon
        icon_offset_y = 30.0
        icon_offset_x = int((icon.get("roi")[1] - (anchor.get('xbr') - anchor.get('xtl'))) * 0.5)
        icon_position = {
            'ytl': int(anchor.get('ytl') - icon.get('roi')[0] - icon_offset_y),
            'xtl': int(anchor.get('xtl') - icon_offset_x),
            'ybr': int(anchor.get('ytl') - icon_offset_y),
            'xbr': int(anchor.get('xtl') + icon.get('roi')[1] - icon_offset_x)
            }
        self.drawIcon(frame, icon, icon_position)
        cv2.rectangle(
            frame,
            (anchor.get('xtl'), anchor.get('ytl')),
            (anchor.get('xbr'), anchor.get('ybr')),
            (0, 255, 0) if self.is_checked else (0, 0, 255),
            2
            )

        # future trajectory
        arrow_color = 'green' if self.is_checked else 'red'
        arrow_info = self.icon_dict.get(f"{database.get('future_direction')}_{arrow_color}")
        arrow_position = {
            'ytl': int(self.video_res[0] - 300),
            'xtl': int(self.video_res[1] / 2 - arrow_info.get('roi')[1]/2),
            'ybr': int(self.video_res[0] - 300 + arrow_info.get('roi')[0]),
            'xbr': int(self.video_res[1] / 2 + arrow_info.get('roi')[1]/2)
        }
        self.drawIcon(frame, arrow_info, arrow_position)

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
        rate = (frame_count - database.get("start_frame")) / (database.get("critical_point") - database.get("start_frame"))
        self.showProgress(frame, icon_position, rate)

    def drawIcon(self, frame, icon_info, position):
        """draw icon to emphasize the target objects
        frame : frame
        position : PIE dataset info of the object in the frame
        """

        if position.get('ytl') < 0 or position.get('ybr') > self.video_res[0] or position.get('xtl') < 0 or position.get('xbr') > self.video_res[1]:
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
        if rate <= 0.4:
            color = (0, 255, 0)
        elif 0.4 < rate <= 0.7:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

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
        ## ellipse probress bar
        # cv2.ellipse(
        # image,
        # ((icon_position.get('xbr') + icon_position.get('xtl')) // 2, (icon_position.get('ybr') + icon_position.get('ytl')) // 2),
        # (max(icon_info.get('roi')) // 2, max(icon_info.get('roi')) // 2),
        # -90, (1 - rate) * 360, 0, color, thickness=3, lineType=cv2.LINE_AA
        # )
        # cv2.line(
        # image,
        # (0, int(self.video_res[0]-8)), (int(self.video_res[1] * (1 - rate)), int(self.video_res[0]-8)),
        # color,
        # thickness=16,
        # lineType=cv2.LINE_AA
        # )

    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))

        self.video.release()
        cv2.destroyAllWindows()

        # with open(self.log_file, 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['display_frame', 'display_time', 'id', 'obj_type', 'prob', 'framein_point', 'frameout_point', 'intervene_type', 'intervene_frame', 'intervene_time', 'intervene_key'])
        #     writer.writerows(self.log)


    def calcAnchor(self, anchor):
        out_anchor = {
            "xbr": int((float(anchor.get("xbr")) - self.hmi_base_anchor.get("xbr")) * (1 / self.hmi_crop_rate)),
            "xtl": int((float(anchor.get("xtl")) - self.hmi_base_anchor.get("xtl")) * (1 / self.hmi_crop_rate)),
            "ybr": int((float(anchor.get("ybr")) - self.hmi_base_anchor.get("ybr")) * (1 / self.hmi_crop_rate)),
            "ytl": int((float(anchor.get("ytl")) - self.hmi_base_anchor.get("ytl")) * (1 / self.hmi_crop_rate)),
            }
        return out_anchor


    def play(self, database):
        self.database = database
        video = self.getVideo(database.get("video_file"))
        video.set(cv2.CAP_PROP_POS_FRAMES, database.get("start_frame"))
        ret, frame = video.read()
        self.frame_count = database.get("start_frame")

        print(self.frame_count, database.get("crossing_point"))
        while ret and database.get("crossing_point") > self.frame_count:
            start = time.time()
            hmi_frame = cv2.resize(frame, dsize=None, fx=1.0/self.max_hmi_crop_rate, fy=1.0/self.max_hmi_crop_rate)
            expand_px = self.renderInfo(hmi_frame, self.database, self.frame_count) # add info to the frame
            cv2.moveWindow("hmi", 10, 0)
            cv2.imshow(
                "hmi",
                hmi_frame[
                    self.hmi_base_anchor.get("ytl")-expand_px:self.hmi_base_anchor.get("ybr")+expand_px,
                    self.hmi_base_anchor.get("xtl")-expand_px:self.hmi_base_anchor.get("xbr")+expand_px
                    ]
                ) # render

            # croped_frame = self.cropFrame(frame, self.windshield_anchor, self.windshield_expand_rate)
            # cv2.imshow("windshield", croped_frame) # render
            # cv2.moveWindow("windshield", 10, 0)
            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (30) - (time.time() - start))), 1)
            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key != 255 : print(key)
            if key == ord('q'):
                exit(1)
            if key == 13 or key == ord('y') or key == ord('n'):
                self.buttonCallback(key)
                # break

            ret, frame = video.read()
            self.frame_count += 1
