#! /usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import pickle
import numpy as np

class PIEVisualize():
    def __init__(self):
        self.hmi_image_offset_y = 0.2
        self.hmi_crop_rate = 0.6
        self.hmi_expand_rate = 1.0 / self.hmi_crop_rate

        self.windshield_crop_size = None
        self.windshield_expand_rate = None

        self.image_res = None
        self.hmi_crop_size = None
        self.video_fps = None

        self.icon_dict = {}

        self.prepareEventHandler()
        self.prepareIcon("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/pie_icons")

    def getVideoInfo(self, video):
        # get video rate and change variable unit from time to frame num
        self.video_fps = int(video.get(cv2.CAP_PROP_FPS))
        self.image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]
        # adjust video rate to keep genuine broadcast rate

        # calc image-crop-region crop -> expaned to original frame geometry
        offset_yt = self.image_res[0] * ((1.0 - self.hmi_crop_rate) * 0.5 + self.hmi_image_offset_y)
        offset_xl = self.image_res[1] * (1.0 - self.hmi_crop_rate) * 0.5
        self.crop_size = [
            int(offset_yt),
            int(offset_yt + self.image_res[0] * self.hmi_crop_rate),
            int(offset_xl),
            int(offset_xl + self.image_res[1] * self.hmi_crop_rate)
            ]


    def prepareIcon(filename):
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
        # if the event handler is leftButtonDown
        if event == cv2.EVENT_LBUTTONDOWN and self.current_obj_info['xtl'] < x < self.current_obj_info['xbr'] and self.current_obj_info['ytl'] < y < self.current_obj_info['ybr']:
            self.log[-1] += [time.time(), 'touched', None]
            self.is_checked = not self.is_checked


    def buttonCallback(self, key):
        """callback of enter key push, target is focused object
        """
        self.log[-1] += [time.time(), 'pushed', key]
        self.is_checked = not self.is_checked


    def renderInfo(self, image, obj_info, frame_count):
        """add information to the image

        """
        self.current_obj_info = obj_info.get('frames_info')[frame_count]
        # print(self.icon_dict)

        if obj_info.get('label') == 'traffic_light':

            icon_info = self.icon_dict.get('tf_green') if self.is_checked else self.icon_dict.get('tf_red')
            icon_offset_y = 30.0
            icon_offset_x = int((icon_info.get('roi')[1] - (self.current_obj_info['xbr'] - self.current_obj_info['xtl'])) * 0.5)

        if obj_info.get('label') == 'pedestrian':
            if self.is_checked:
                icon_info = self.icon_dict.get('walker_checked')
            else:
                if self.current_obj_info.get('xbr') < self.image_res[1] * 0.5:
                    icon_info = self.icon_dict['walker_cross_to_right']
                else:
                    icon_info = self.icon_dict['walker_cross_to_left']

            icon_offset_y = 30.0
            icon_offset_x = int((icon_info.get('roi')[1] - (self.current_obj_info.get('xbr') - self.current_obj_info.get('xtl'))) * 0.5)

        # position of the icon
        icon_position = {
            'ytl': int(self.current_obj_info.get('ytl') - icon_info.get('roi')[0] - icon_offset_y),
            'xtl': int(self.current_obj_info.get('xtl') - icon_offset_x),
            'ybr': int(self.current_obj_info.get('ytl') - icon_offset_y),
            'xbr': int(self.current_obj_info.get('xtl') + icon_info.get('roi')[1] - icon_offset_x)
            }

        # print(float(self.current_obj_info.get('yaw')))
        # print(float(obj_info.get('frames_info')[-1].get('heading_angle')) - float(self.current_obj_info.get('heading_angle')))
        self.drawIcon(image, icon_info, icon_position, self.image_res)

        arrow_color = 'green' if self.is_checked else 'red'
        arrow_info = self.icon_dict.get(f"{obj_info.get('future_direction')}_{arrow_color}")
        arrow_position = {
            'ytl': int(self.image_res[0] - 300),
            'xtl': int(self.image_res[1] / 2 - arrow_info.get('roi')[1]/2),
            'ybr': int(self.image_res[0] - 300 + arrow_info.get('roi')[0]),
            'xbr': int(self.image_res[1] / 2 + arrow_info.get('roi')[1]/2)
        }
        self.drawIcon(image, arrow_info, arrow_position, self.image_res)

        color = (0, 255, 0) if self.is_checked else (0, 0, 255)

        # cv2.putText(
        #     image,
        #     '{:.01f}'.format(obj_info['prob']),
        #     # '{:.01f}s'.format((obj_info['critical_point'] - obj_info['start_point'] - frame_count) / self.fps),
        #     (self.current_obj_info['xtl'], self.current_obj_info['ytl'] + 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, color, 3, cv2.LINE_AA
        #     )
        lest_time = frame_count / (obj_info['critical_point'] - obj_info['start_point'])
        if lest_time <= 0.4:
            progress_color = (0, 255, 0)
        elif 0.4 < lest_time <= 0.7:
            progress_color = (0, 255, 255)
        else:
            progress_color = (0, 0, 255)

        # cv2.ellipse(
        # image,
        # ((icon_position.get('xbr') + icon_position.get('xtl')) // 2, (icon_position.get('ybr') + icon_position.get('ytl')) // 2),
        # (max(icon_info.get('roi')) // 2, max(icon_info.get('roi')) // 2),
        # -90, (1 - lest_time) * 360, 0, progress_color, thickness=3, lineType=cv2.LINE_AA
        # )
        # cv2.line(
        # image,
        # (0, int(self.image_res[0]-8)), (int(self.image_res[1] * (1 - lest_time)), int(self.image_res[0]-8)),
        # progress_color,
        # thickness=16,
        # lineType=cv2.LINE_AA
        # )
        cv2.line(
            image,
            (icon_position.get('xtl'), icon_position.get('ybr') + 10),
            (icon_position.get('xbr'), icon_position.get('ybr') + 10),
            (0, 0, 0),
            thickness=8,
            lineType=cv2.LINE_AA
            )
        cv2.line(
            image,
            (icon_position.get('xtl'), icon_position.get('ybr') + 10),
            ( icon_position.get('xtl') + int((icon_position.get('xbr') - icon_position.get('xtl')) * (1 - lest_time)), icon_position.get('ybr') + 10),
            progress_color,
            thickness=8,
            lineType=cv2.LINE_AA
            )

        image = cv2.rectangle(
            image,
            (self.current_obj_info['xtl'], self.current_obj_info['ytl']),
            (self.current_obj_info['xbr'], self.current_obj_info['ybr']),
            color, 2
            )


    def drawIcon(image, icon_info, position, image_res):
        """draw icon to emphasize the target objects
        image : image
        position : PIE dataset info of the object in the frame
        """

        if position.get('ytl') < 0 or position.get('ybr') > image_res[0] or position.get('xtl') < 0 or position.get('xbr') > image_res[1]:
            print(f"icon is out of range y:{position.get('ytl')}-{position.get('ybr')}, x:{position.get('xtl')}-{position.get('xbr')}")
            return

        # put icon on image
        try:
            roi = image[position.get('ytl'):position.get('ybr'), position.get('xtl'):position.get('xbr')] # get roi from image
            image_bg = cv2.bitwise_and(roi, roi, mask=icon_info['mask_inv']) # remove color from area for icon by filter
            buf = cv2.add(icon_info['icon_fg'], image_bg) # put icon of roi
            image[position.get('ytl'):position.get('ybr'), position.get('xtl'):position.get('xbr')] = buf # replace image region to roi

        except Exception as e:
            print(f'failed to put icon: {e}')


    def calcAnchor(self, anchor, crop_size, crop_rate):
        out_anchor = []
        anchor = [
            int((float(anchor[0]) - crop_size[2]) * (1 / crop_rate)),
            int((float(anchor[1]) - crop_size[2]) * (1 / crop_rate)),
            int((float(anchor[2]) - crop_size[0]) * (1 / crop_rate)),
            int((float(anchor[3]) - crop_size[0]) * (1 / crop_rate)),
            ]
        return out_anchor


    def play(self, video, databese):
        video.set(cv2.CAP_PROP_POS_FRAMES, databese.get("start_point"))
        ret, frame = video.read()
        frame_count = databese.get("start_point")

        while ret and databese.get("crossing_point") > frame_count:
            start = time.time()

            croped_frame = self.cropFrame(frame, self.hmi_crop_size, self.hmi_expand_rate)
            rendered_frame = self.renderInfo(croped_frame, databese, frame_count) # add info to the frame
            cv2.namedWindow("hmi", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("hmi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("hmi", rendered_frame) # render
            cv2.moveWindow("hmi", 10, 0)

            croped_frame = self.cropFrame(frame, self.windshield_crop_size, self.windshield_expand_rate)
            cv2.namedWindow("windshield", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("windshield", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("windshield", croped_frame) # render
            cv2.moveWindow("windshield", 10, 0)
            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (self.modified_fps) - (time.time() - start))), 1)
            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key is not 255 : print(key)
            if key == ord('q'):
                exit(1)
            if key == 13 or key == ord('y') or key == ord('n'):
                self.buttonCallback(key)
                # break

            ret, frame = video.read()
            frame_count += 1
