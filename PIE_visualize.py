#! /usr/bin/python3
# -*- coding:utf-8 -*-

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

        # static variables calcurated in this class
        self.image_res = None
        # self.image_res = args.res
        self.modified_video_rate = None
        self.window_name = 'frame'

        # dynamic object
        self.video = None
        self.current_frame_num = None
        self.current_obj_dict = {}


    def __enter__(self):
        return self


    def getVideo(self, args):

        try:
            self.video = cv2.VideoCapture(args.video)

        except:
            print('cannot open video')
            exit(0)

        # get video rate and change variable unit from time to frame num
        video_rate = int(self.video.get(cv2.CAP_PROP_FPS))
        self.image_res = [self.vedeo.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(cv2.CAP_PROP_FRAME_WIDTH)]


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
            frameout_point = None
            tr_blue_prob = random.random()

            for anno_itr in track.iter('box'):
                anno_info = {}

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
                if frameout_point is None:
                    # if the pedestrian framed out now, save framenum and apply it to all obj with same id
                    if anno_info['xbr'] > self.image_res[1] or anno_info['xtl'] < 0 or anno_info['ybr'] > self.image_res[0] or anno_info['ytl'] < 0:
                        frameout_point = anno_itr.attrib.get('frame')
                        anno_info['frameout_point'] = frameout_point

                        # scan the already added object in self.pie_data[]
                        for frame_obj in self.pie_data.values():
                            for obj_id, obj_info in frame_obj.items():
                                if obj_id == anno_info['id']:
                                    obj_info['frameout_point'] = frameout_point

                    # if the framedout_point is not found, the framedout_point filled with final annotated frame
                    else:
                        anno_info['frameout_point'] = track.findall('box')[-1].attrib.get('frame')

                else:
                    anno_info['frameout_point'] = frameout_point

                # if object is pedestrian, get additional information from attributes.xml
                if anno_info['label'] == 'pedestrian':
                    for attrib_itr in self.attrib_tree.iter('pedestrian'):
                        if attrib_itr.attrib.get('id') == anno_info['id']:
                            anno_info['prob'] = float(attrib_itr.attrib.get('intention_prob'))
                            anno_info['critical_point'] = attrib_itr.attrib.get('critical_point')
                            anno_info['crossing_point'] = attrib_itr.attrib.get('crossing_point')
                            anno_info['exp_start_point'] = attrib_itr.attrib.get('exp_start_point')

                # if object is trafficlight, mimic pedestrian and interporate additional information
                if anno_info['label'] == 'traffic_light':
                    anno_info['prob'] = tr_blue_prob
                    anno_info['critical_point'] = anno_info['frameout_point']
                    anno_info['crossing_point'] = anno_info['frameout_point']
                    anno_info['exp_start_point'] = track[0].attrib.get('frame') # appear frame

                # add to pie_data dictionary
                if anno_itr.attrib.get('frame') not in self.pie_data:
                    self.pie_data[anno_itr.attrib.get('frame')] = {}

                self.pie_data[anno_itr.attrib.get('frame')][anno_info['id']] = anno_info

        # delete objects to improve performance
        del root
        del tree
        del self.attrib_tree


    def refleshCurrentObj(self):
        """magage displaying object
        is_checked ; flag wether the subject check obj and input some action or not
        """
        latest_obj = {} # new container
        is_forcused_obj_exist = False # flag

        # judge whether the object should be displayed
        for obj_id, obj_info in self.pie_data[self.current_frame_num].items():
            if obj_info['label'] not in  ['pedestrian', 'traffic_light']: continue
            if obj_info['label'] == 'pedestrian' and obj_info['prob'] > self.prob_thres_pedestrian: continue
            if obj_info['label'] == 'traffic_light':
                if obj_info['type'] != 'regular': continue
                if obj_info['xbr'] < self.image_res[1] * 0.5: continue # remove opposite side light
                if obj_info['prob'] > self.prob_thres_tr: continue


            latest_obj[obj_id] = {
                'is_spawned_range': int(obj_info['exp_start_point']) <= int(self.current_frame_num) < int(obj_info['frameout_point']),
                'is_passed': int(self.current_frame_num) > int(obj_info['frameout_point']),
                'is_ready': int(obj_info['exp_start_point']) > int(self.current_frame_num)
            }

        self.current_obj_dict = latest_obj


def renderInfo(self, image, current_obj_dict, current_frame_num):
    """add information to the image

    """
    # loop for each object in the frame from PIE dataset
    for obj_id, displayed_obj_info in current_obj_dict.items():
        obj_info = pie_data[current_frame_num][obj_id]

        if displayed_obj_info['is_focused']: # if forcused --- red
            color = (0, 0, 255)

        elif displayed_obj_info['is_passed']: # if checked --- green
            color = (0, 255, 0)

        elif displayed_obj_info['is_ready']: # passed --- blue
            color = (255, 0, 0)

        else:
            color = (0,0,0)

        image = cv2.rectangle(image,
        (obj_info['xtl'], obj_info['ytl']),
        (obj_info['xbr'], obj_info['ybr']),
        color,
        1)


def loop(pie_data_visualize):

    print('start_loop')

    sleep_time = pie_data_visualize.modified_video_rate
    frame = 0

    while(pie_data_visualize.video.isOpened()):
        start = time.time()
        pie_data_visualize.current_frame_num = str(frame)
        ret, image = pie_data_visualize.video.read()

        if pie_data_visualize.current_frame_num in pie_data_visualize.pie_data:
            pie_data_visualize.updateFocusedObject() # udpate pie_data_visualize.current_obj_dict
            pie_data_visualize.renderInfo(image) # add info to the image

        cv2.namedWindow(pie_data_visualize.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(pie_data_visualize.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(pie_data_visualize.window_name, image) # render
        cv2.moveWindow(pie_data_visualize.window_name, 10, 0)
        #  calc sleep time to keep frame rate to be same with video rate
        sleep_time = max(int((1000 / (pie_data_visualize.modified_video_rate) - (time.time() - start))), 1)

        frame += 1

    exit(1)


    def __exit__(self, exc_type, exc_value, traceback):
        print('delete instance... type: {}, value: {}, traceback: {}'.format(exc_type, exc_value, traceback))

        self.video.release()
        cv2.destroyAllWindows()


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
    args = argparser.parse_args()

    with PieDataVisualize(args) as pie_data_visualize:
        pie_data_visualize.getVideo(args)
        pie_data_visualize.getAttrib(args.attrib)
        pie_data_visualize.getAnno(args.anno)
        loop(pie_data_visualize)

if __name__ == '__main__':
    main()
