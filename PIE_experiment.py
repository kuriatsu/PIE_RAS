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
import xml.etree.ElementTree as ET
import random


class PieDataVisualize(object):

    def __init__(self, args):
        # static object
        self.pie_data = {} # obj info of each frames
        self.attrib_tree = None

        self.icon_dict = {
            'walker_cross_to_left':
                {
                'path':args.icon_path + 'walker_cross_to_left.png'
                },
            'walker_cross_to_right':
                {
                'path':args.icon_path + 'walker_cross_to_right.png'
                },
            'tf_red':
                {
                'path':args.icon_path + 'tf_red.png'
                },
            'tf_green':
                {
                'path':args.icon_path + 'tf_green.png'
                }
            }

        # static variables calcurated in this class
        self.image_res=None
        self.window_position=args.window_position
        self.video_rate = None

        # self.image_res = args.res
        self.modified_video_rate = None
        self.image_crop_rate = args.image_crop_rate
        self.window_name = 'frame'
        self.prob_thres_tr = args.prob_thres_tr
        self.prob_thres_pedestrian = args.prob_thres_pedestrian
        self.obj_spawn_time_min = args.obj_spawn_time_min
        # log
        self.log_file = args.log
        self.log = []

        # dynamic object
        self.video = None
        self.current_frame_num = None
        self.focused_obj_id = None


    def __enter__(self):
        return self


    def prepareIcon(self):

        for icon_info in self.icon_dict.values():
            img = cv2.imread(icon_info.get('path'))
            icon_info['roi'] = img.shape[:2]
            img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
            # ret, mask = cv2.threshold(img2grey, 200, 255, cv2.THRESH_BINARY_INV)
            icon_info['mask_inv'] = cv2.bitwise_not(mask)
            icon_info['icon_fg'] = cv2.bitwise_and(img, img, mask=mask)


    def getVideo(self, args):

        try:
            self.video = cv2.VideoCapture(args.video)

        except:
            print('cannot open video')
            exit(0)

        # get video rate and change variable unit from time to frame num
        self.video_rate = int(self.video.get(cv2.CAP_PROP_FPS))
        self.image_res = [self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(cv2.CAP_PROP_FRAME_WIDTH)]
        # adjust video rate to keep genuine broadcast rate
        self.modified_video_rate = self.video_rate + args.rate_offset

        # calc image-crop-region crop -> expaned to original frame geometry
        offset_yt = self.image_res[0] * ((1.0 - self.image_crop_rate) * 0.5 + args.image_crop_offset_y)
        offset_xl = self.image_res[1] * (1.0 - self.image_crop_rate) * 0.5
        self.image_offset = [int(offset_yt),
                             int(offset_yt + self.image_res[0] * self.image_crop_rate),
                             int(offset_xl),
                             int(offset_xl + self.image_res[1] * self.image_crop_rate)
                             ]


    def getAttrib(self, attrib_file):
        try:
            tree = ET.parse(attrib_file)
            self.attrib_tree = tree.getroot()
            del tree
        except:
            print('cannot open attrib file')
            exit(0)


    def getAnno(self, anno_file):
        try:
            tree = ET.parse(anno_file)
            root = tree.getroot()
        except:
            print('cannot open annotation file')
            exit(0)


        for track in root.findall('track'):
            tr_blue_prob = random.random()
            framein_point = None
            frameout_point = None
            id = None

            for anno_itr in track.iter('box'):
                anno_info = {}
                anno_frame = int(anno_itr.attrib.get('frame'))
                # get id and other info from child tree under <box>
                for attribute in anno_itr.findall('attribute'):
                    if attribute.attrib.get('name') in ['id', 'cross', 'type', 'state']:
                        anno_info[attribute.attrib.get('name')] = attribute.text

                # get basic information
                anno_info['label'] = track.attrib.get('label')
                anno_info['xbr'] = int((float(anno_itr.attrib.get('xbr')) - self.image_offset[2]) * (1 / self.image_crop_rate))
                anno_info['xtl'] = int((float(anno_itr.attrib.get('xtl')) - self.image_offset[2]) * (1 / self.image_crop_rate))
                anno_info['ybr'] = int((float(anno_itr.attrib.get('ybr')) - self.image_offset[0]) * (1 / self.image_crop_rate))
                anno_info['ytl'] = int((float(anno_itr.attrib.get('ytl')) - self.image_offset[0]) * (1 / self.image_crop_rate))
                anno_info['size'] = (anno_info['xbr'] - anno_info['xtl']) * (anno_info['ybr'] - anno_info['ytl'])

                # if the object frameded out, save the frame num and apply it to the already added data in self.pie_data[]
                if anno_info['xbr'] > self.image_res[1] or anno_info['xtl'] < 0 or anno_info['ybr'] > self.image_res[0] or anno_info['ytl'] < 0:
                    if framein_point is not None and frameout_point is None:
                        frameout_point = anno_frame

                else:
                    if framein_point is None:
                        framein_point = anno_frame
                        id = anno_info['id']

                # if object is pedestrian, get additional information from attributes.xml
                if anno_info['label'] == 'pedestrian':
                    for attrib_itr in self.attrib_tree.iter('pedestrian'):
                        if attrib_itr.attrib.get('id') == anno_info['id']:
                            anno_info['prob'] = float(attrib_itr.attrib.get('intention_prob'))
                            anno_info['critical_point'] = int(attrib_itr.attrib.get('critical_point'))
                            anno_info['crossing_point'] = int(attrib_itr.attrib.get('crossing_point'))
                            anno_info['exp_start_point'] = int(attrib_itr.attrib.get('exp_start_point'))

                # if object is trafficlight, mimic pedestrian and interporate additional information
                if anno_info['label'] == 'traffic_light':
                    anno_info['prob'] = tr_blue_prob
                    anno_info['critical_point'] = None
                    anno_info['crossing_point'] = int(track[-1].attrib.get('frame'))
                    anno_info['exp_start_point'] = anno_frame # appear frame

                # add to pie_data dictionary
                if anno_frame not in self.pie_data:
                    self.pie_data[anno_frame] = {}

                self.pie_data[anno_frame][anno_info['id']] = anno_info

            if frameout_point is None:
                frameout_point = int(track[-1].attrib.get('frame'))

            if framein_point is not None:
                for i in range(framein_point, frameout_point):
                    try:
                        self.pie_data[i][id]['framein_point'] = framein_point
                        self.pie_data[i][id]['frameout_point'] = frameout_point
                    except KeyError:
                        pass

        # delete objects to improve performance
        del root
        del tree
        del self.attrib_tree


    def prepareEventHandler(self):
        """add mouse click callback to the window
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.touchCallback)


    def touchCallback(self, event, x, y, flags, param):
        """if mouce clicked, check position and judge weather the position is on the rectange or not
        """
        if self.focused_obj_id is not None:
            focused_obj_info = self.pie_data[self.current_frame_num][self.focused_obj_id]
            # if the event handler is leftButtonDown
            if event == cv2.EVENT_LBUTTONDOWN and focused_obj_info['xtl'] < x < focused_obj_info['xbr'] and focused_obj_info['ytl'] < y < focused_obj_info['ybr']:
                self.focused_obj_id = None
                self.log[-1] += ['touched' ,self.current_frame_num, time.time()]


    def pushCallback(self, key):
        """callback of enter key push, target is focused object
        """
        if self.focused_obj_id is not None:
            self.focused_obj_id = None
            self.log[-1] += ['pushed', self.current_frame_num, time.time(), key]


    def updateFocusedObject(self):
        """find focused object from self.target_obj_dict
        """
        # print('focused_obj', self.focused_obj_id)
        if self.focused_obj_id is not None:
            if self.pie_data[self.current_frame_num][self.focused_obj_id]['critical_point'] < self.current_frame_num:
                self.log[-1] += ['passed', self.current_frame_num, time.time()]
                self.focused_obj_id = None

        # is this method called by callback, checked_obj_id has target object id which is checked and should be unfocused
        if self.focused_obj_id is None:
            # find new focused object
            for obj_id, obj_info in self.pie_data[self.current_frame_num].items():
                if obj_info['label'] not in  ['pedestrian', 'traffic_light']: continue
                if 'framein_point' not in obj_info and 'frameout_point' not in obj_info: continue
                if obj_info['frameout_point'] - obj_info['framein_point'] < self.obj_spawn_time_min: continue
                if self.current_frame_num < obj_info['framein_point'] or obj_info['frameout_point'] < self.current_frame_num: continue

                if obj_info['exp_start_point'] == self.current_frame_num:
                    if obj_info['label'] == 'pedestrian':
                        if obj_info['prob'] > self.prob_thres_pedestrian: continue

                    if obj_info['label'] == 'traffic_light':
                        if obj_info['type'] != 'regular': continue
                        if obj_info['xbr'] < self.image_res[1] * 0.5: continue
                        if obj_info['prob'] > self.prob_thres_tr: continue

                        min_next_pedestrian_frame = 90
                        for obj_info_for_pedes in self.pie_data[self.current_frame_num].values():
                            if obj_info_for_pedes['label'] == 'pedestrian':
                                next_pedesrian_frame = obj_info_for_pedes['exp_start_point'] - self.current_frame_num
                                if min_next_pedestrian_frame > next_pedesrian_frame > 0:
                                    min_next_pedestrian_frame = next_pedesrian_frame

                        if min_next_pedestrian_frame < 90: continue

                        for i in range(self.current_frame_num, obj_info['crossing_point']+1):
                            if i in self.pie_data:
                                if obj_id in self.pie_data[i]:
                                    self.pie_data[i][obj_id]['critical_point'] = min(self.current_frame_num + 90, obj_info['frameout_point'])
                                    self.pie_data[i][obj_id]['exp_start_point'] = self.current_frame_num

                    self.focused_obj_id = obj_id
                    self.log.append([
                        self.current_frame_num,
                        time.time(),
                        obj_id,
                        obj_info['label'],
                        obj_info['prob'],
                        obj_info['framein_point'],
                        obj_info['frameout_point']
                        ])

                    return


    def renderInfo(self, image):
        """add information to the image

        """
        if self.focused_obj_id is None: return
        focused_obj_info = self.pie_data[self.current_frame_num][self.focused_obj_id]

        self.drawIcon(image, focused_obj_info)

        color = (0, 0, 255)
        # cv2.putText(
        #     image,
        #     # 'Cross?',
        #     # '{:.01f}'.format((focused_obj_info['xbr'] - focused_obj_info['xtl']) * (focused_obj_info['ybr'] - focused_obj_info['ytl'])),
        #     '{:.01f}%'.format(focused_obj_info['prob'] * 100),
        #     # '{:.01f}s'.format((focused_obj_info['critical_point'] - self.current_frame_num) / self.video_rate),
        #     (int(focused_obj_info['xtl']), int(focused_obj_info['ytl']) - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA
        #     )

        image = cv2.rectangle(image,
        (focused_obj_info['xtl'], focused_obj_info['ytl']),
        (focused_obj_info['xbr'], focused_obj_info['ybr']),
        color,
        1)


    def drawIcon(self, image, obj_info):
        """draw icon to emphasize the target objects
        image : image
        obj_info : PIE dataset info of the object in the frame
        """

        if obj_info['label'] == 'traffic_light':
            icon_info = self.icon_dict['tf_red']
            icon_offset_y = 30.0
            icon_offset_x = int((icon_info['roi'][1] - (obj_info['xbr'] - obj_info['xtl'])) * 0.5)


        if obj_info['label'] == 'pedestrian':
            if obj_info['xbr'] < self.image_res[1] * 0.5:
                icon_info = self.icon_dict['walker_cross_to_right']
            else:
                icon_info = self.icon_dict['walker_cross_to_left']

            icon_offset_y = 30.0
            icon_offset_x = int((icon_info['roi'][1] - (obj_info['xbr'] - obj_info['xtl'])) * 0.5)

        # position of the icon
        icon_ytl = int(obj_info['ytl'] - icon_info['roi'][0] - icon_offset_y)
        icon_xtl = int(obj_info['xtl'] - icon_offset_x)
        icon_ybr = int(obj_info['ytl'] - icon_offset_y)
        icon_xbr = int(obj_info['xtl'] + icon_info['roi'][1] - icon_offset_x)

        # put icon on image
        try:
            roi = image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] # get roi from image
            image_bg = cv2.bitwise_and(roi, roi, mask=icon_info['mask_inv']) # remove color from area for icon by filter
            buf = cv2.add(icon_info['icon_fg'], image_bg) # put icon of roi
            image[icon_ytl:icon_ybr, icon_xtl:icon_xbr] = buf # replace image region to roi

        except:
            print('icon is out of range y:{}-{}, x:{}-{}'.format(icon_ytl, icon_ybr, icon_xtl, icon_xbr))


    def loop(self):

        print('start_loop')

        sleep_time = self.modified_video_rate
        frame = 0

        while(self.video.isOpened()):
            start = time.time()
            self.current_frame_num = frame
            ret, image = self.video.read()


            # preprocess image . crop + resize
            scale = 1.0 / self.image_crop_rate
            image = cv2.resize(image[self.image_offset[0]:self.image_offset[1], self.image_offset[2]:self.image_offset[3]], dsize=None, fx=scale, fy=scale)

            if self.current_frame_num in self.pie_data:
                self.updateFocusedObject() # udpate self.target_obj_dict
                self.renderInfo(image) # add info to the image

            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.window_name, image) # render
            cv2.moveWindow(self.window_name, int(self.window_position[0]), int(self.window_position[1]))
            #  calc sleep time to keep frame rate to be same with video rate
            sleep_time = max(int((1000 / (self.modified_video_rate) - (time.time() - start))), 1)

            # sleep and wait quit key
            key = cv2.waitKey(sleep_time) & 0xFF
            if key is not 255 : print(key)
            if key == ord('q'):
                break
            if key == 13 or key == ord('y') or key == ord('n'):
                self.pushCallback(key)

            frame += 1

        exit(1)


    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))

        self.video.release()
        cv2.destroyAllWindows()

        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['display_frame', 'display_time', 'id', 'obj_type', 'prob', 'framein_point', 'frameout_point', 'intervene_type', 'intervene_frame', 'intervene_time', 'intervene_key'])
            writer.writerows(self.log)


def main():
    parser_res = lambda x : list(map(float, x.split('x')))

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '--video', '-v',
        metavar='VIDEO',
        default='/media/ssd/PIE_data/PIE_clips/set02/video_0001.mp4')
    argparser.add_argument(
        '--anno',
        metavar='ANNO',
        default='/media/ssd/PIE_data/annotations/set02/video_0001_annt.xml')
    argparser.add_argument(
        '--attrib',
        metavar='ATTRIB',
        default='/media/ssd/PIE_data/annotations_attributes/set02/video_0001_attributes.xml')
    argparser.add_argument(
        '--rate_offset',
        metavar='OFFSET',
        default=15)
    argparser.add_argument(
        '--log',
        metavar='LOG',
        default='/home/kuriatsu/share/{}_intervene_time.csv'.format(datetime.date.today()))
    argparser.add_argument(
        '--image_crop_rate',
        metavar='SCALE',
        default=0.6)
    argparser.add_argument(
        '--image_crop_offset_y',
        metavar='OFFSET',
        default=0.2)
    argparser.add_argument(
        '--res',
        metavar='height x width',
        type=parser_res,
        default='1080x1900')
    argparser.add_argument(
        '--window_position',
        metavar='x x y',
        type=parser_res,
        default='10x0')
    argparser.add_argument(
        '--obj_spawn_time_min',
        metavar='MIN_TIME',
        default=0.5)
    argparser.add_argument(
        '--icon_path',
        metavar='/path/to/icon/files',
        default='/home/kuriatsu/share/')
    argparser.add_argument(
        '--prob_thres_pedestrian',
        metavar='RATE',
        default=1.0)
    argparser.add_argument(
        '--prob_thres_tr',
        metavar='RATE',
        default=0.0)
    args = argparser.parse_args()

    with PieDataVisualize(args) as pie_data_visualize:
        pie_data_visualize.getVideo(args)
        pie_data_visualize.getAttrib(args.attrib)
        pie_data_visualize.getAnno(args.anno)
        pie_data_visualize.prepareIcon()
        pie_data_visualize.prepareEventHandler()
        pie_data_visualize.loop()

if __name__ == '__main__':
    main()
