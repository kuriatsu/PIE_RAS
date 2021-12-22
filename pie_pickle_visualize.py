#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
welcome to PIE data visualize

"""

import cv2
import argparse
import numpy as np
import time
import datetime
import csv
import pickle
import random
from pie_lib import PieLib


class PiePickleVisualize(object):

    def __init__(self):
        # static object set at main()
        self.icon_dict = None
        self.image_res=None
        self.window_position = None
        self.fps = None
        self.modified_fps = None
        self.prob_thres_tr = args.prob_thres_tr
        self.prob_thres_pedestrian = args.prob_thres_pedestrian
        self.log_file = None

        self.pie_data = None
        self.window_name = 'frame'
        self.log = []

        self.is_checked = False
        self.current_obj_info = None
        # log


    def __enter__(self):
        return self


    def prepareEventHandler(self):
        """add mouse click callback to the window
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.touchCallback)


    def touchCallback(self, event, x, y, flags, param):
        """if mouce clicked, check position and judge weather the position is on the rectange or not
        """
        # if the event handler is leftButtonDown
        if event == cv2.EVENT_LBUTTONDOWN and self.current_obj_info['xtl'] < x < self.current_obj_info['xbr'] and self.current_obj_info['ytl'] < y < self.current_obj_info['ybr']:
            self.log[-1] += [time.time(), 'touched', None]
            self.is_checked = not self.is_checked


    def pushCallback(self, key):
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
        PieLib.drawIcon(image, icon_info, icon_position, self.image_res)

        arrow_color = 'green' if self.is_checked else 'red'
        arrow_info = self.icon_dict.get(f"{obj_info.get('future_direction')}_{arrow_color}")
        arrow_position = {
            'ytl': int(self.image_res[0] - 300),
            'xtl': int(self.image_res[1] / 2 - arrow_info.get('roi')[1]/2),
            'ybr': int(self.image_res[0] - 300 + arrow_info.get('roi')[0]),
            'xbr': int(self.image_res[1] / 2 + arrow_info.get('roi')[1]/2)
        }
        PieLib.drawIcon(image, arrow_info, arrow_position, self.image_res)

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


    def loop(self, video, id, obj_info):

        print(f"start_loop, frame_len:{len(obj_info.get('frames_info'))}")

        frame_count = 0
        initial_frame_info = obj_info.get('frames_info')[0]
        self.log.append([
            id,
            time.time(),
            obj_info['label'],
            obj_info['prob'],
            initial_frame_info.get('distance', None),
            initial_frame_info.get('speed', None),
            np.sqrt((initial_frame_info.get('xbr') - initial_frame_info.get('xtl')) ** 2 + (initial_frame_info.get('ybr') - initial_frame_info.get('ytl')) ** 2)],
            )

        ret, frame = video.read()

        while ret and len(obj_info.get('frames_info')) > frame_count:
            start = time.time()
            self.renderInfo(frame, obj_info, frame_count) # add info to the frame
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.window_name, frame) # render
            cv2.moveWindow(self.window_name, int(self.window_position[0]), int(self.window_position[1]))

            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (self.modified_fps) - (time.time() - start))), 1)
            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key is not 255 : print(key)
            if key == ord('q'):
                exit(1)
            if key == 13 or key == ord('y') or key == ord('n'):
                self.pushCallback(key)
                # break

            ret, frame = video.read()
            frame_count += 1

        # exit(1)


    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))

        cv2.destroyAllWindows()

        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'display_time', 'obj_label', 'prob', 'dist_display', 'speed_display', 'size_display', 'intervene_time', 'intervene_type', 'intervene_key'])
            writer.writerows(self.log)


def main(args):
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)

    start_time = time.time()

    with PiePickleVisualize() as pie_pickle_visualize:
        pie_pickle_visualize.icon_dict = PieLib.prepareIcon(args.icon_path)
        pie_pickle_visualize.prepareEventHandler()

        pie_pickle_visualize.window_position = args.window_position
        pie_pickle_visualize.prob_thres_tr = args.prob_thres_tr
        pie_pickle_visualize.prob_thres_pedestrian = args.prob_thres_pedestrian
        pie_pickle_visualize.log_file = args.log_file

        while (time.time() - start_time) < 600:
            id, val = random.choice(list(data.items()))
            print(id)
            pie_pickle_visualize.is_checked = False
            cv_video, pie_pickle_visualize.image_res, pie_pickle_visualize.fps, _ = PieLib.getVideo(f'{args.video}/{id}.mp4', 0, 0)
            pie_pickle_visualize.modified_fps = args.rate_offset + pie_pickle_visualize.fps
            pie_pickle_visualize.loop(cv_video, id, val)
            del data[id]
            cv_video.release()

        print('time is over')


if __name__ == '__main__':

    parser_res = lambda x : list(map(float, x.split('x')))

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '--video', '-v',
        metavar='VIDEO',
        default='/media/ssd/PIE_data/extracted_data/clips')
    argparser.add_argument(
        '--pickle',
        metavar='PICKLE',
        default='/media/ssd/PIE_data/extracted_data/data.pickle')
    argparser.add_argument(
        '--rate_offset',
        metavar='OFFSET',
        default=15)
    argparser.add_argument(
        '--log_file',
        metavar='LOG',
        default='/home/kuriatsu/share/{}_intervene_time.csv'.format(datetime.date.today()))
    argparser.add_argument(
        '--window_position',
        metavar='x x y',
        type=parser_res,
        default='10x0')
    argparser.add_argument(
        '--exp_time',
        metavar='MIN',
        default=10)
    argparser.add_argument(
        '--icon_path',
        metavar='/path/to/icon/files',
        default='/home/kuriatsu/share/PIE_icons')
    argparser.add_argument(
        '--prob_thres_pedestrian',
        metavar='RATE',
        default=1.0)
    argparser.add_argument(
        '--prob_thres_tr',
        metavar='RATE',
        default=0.0)
    args = argparser.parse_args()

    main(args)
