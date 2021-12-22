#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pie_lib import PieLib
import glob
import os
import random
import pickle
import argparse


def pieExtractClip(video, crop_value, crop_rate, attrib_tree, annt_tree, vehicle_tree, out_data, out_file):

    image_res = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))]
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    expand_rate = 1.0 / crop_rate

    print('read attrib')
    for attrib in attrib_tree.iter('pedestrian'):
        out_data[attrib.attrib.get('id')] = {
            'label' : 'pedestrian',
            'prob' : float(attrib.attrib.get('intention_prob')),
            'critical_point' : int(attrib.attrib.get('critical_point')),
            'crossing_point' : int(attrib.attrib.get('crossing_point')),
            'start_point' : int(attrib.attrib.get('exp_start_point'))
            }

    print('read track')
    for track in annt_tree.iter('track'):
        label = track.attrib.get('label')

        if label not in ['pedestrian', 'traffic_light']: continue

        for annt_attrib in track[0].findall('attribute'):
            if annt_attrib.attrib.get('name') == 'id':
                id = annt_attrib.text
                print(id)

        if label == 'traffic_light':
            # remove contraflow light
            if float(track[0].attrib.get('xbr')) < image_res[1] / 2:
                continue
            else:
                out_data[id] = {
                    'label' : label,
                    'prob' : random.random(),
                    'critical_point' : int(track[-1].attrib.get('frame')),
                    'crossing_point' : int(track[-1].attrib.get('frame')),
                    'start_point' : int(float(track[-1].attrib.get('frame')) - random.random() * 3.0 * frame_rate)
                }

        dist_buf = 0

        for i in range(int(track[0].attrib.get('frame')), len(vehicle_tree)):
            dist_buf += float(vehicle_tree[i].attrib.get('OBD_speed')) * 0.03 / 3.6
            future_angle = float(vehicle_tree[i].attrib.get('yaw')) - float(vehicle_tree[int(track[0].attrib.get('frame'))].attrib.get('yaw'))
            if dist_buf > 50:
                break

        if 1.0 < abs(future_angle) % 3.14 < 2.0:
            if future_angle > 0.0:
                out_data.get(id)['future_direction'] = 'right'
            else:
                out_data.get(id)['future_direction'] = 'left'
                direction = -1
        else:
            out_data.get(id)['future_direction'] = 'straight'

        frame_info_list = []
        for annt_itr in track.iter('box'):
            frame_info = {}
            frame_index = int(annt_itr.attrib.get('frame'))

            if frame_index < out_data.get(id).get('start_point'):
                continue

            elif out_data.get(id).get('critical_point') < frame_index:
                break

            frame_info['xbr'] = int((float(annt_itr.attrib.get('xbr')) - crop_value[2]) * (1 / crop_rate))
            frame_info['xtl'] = int((float(annt_itr.attrib.get('xtl')) - crop_value[2]) * (1 / crop_rate))
            frame_info['ybr'] = int((float(annt_itr.attrib.get('ybr')) - crop_value[0]) * (1 / crop_rate))
            frame_info['ytl'] = int((float(annt_itr.attrib.get('ytl')) - crop_value[0]) * (1 / crop_rate))

            frame_info['speed'] = float(vehicle_tree[frame_index].attrib.get('GPS_speed'))


            frame_info['yaw'] = float(vehicle_tree[min(frame_index + 150, len(vehicle_tree)-1)].attrib.get('yaw')) - float(vehicle_tree[frame_index].attrib.get('yaw'))

            if label == 'pedestrian':
                dist = 0

                for vehicle in vehicle_tree[frame_index:out_data.get(id).get('crossing_point')]:
                    dist += float(vehicle.attrib.get('GPS_speed')) * 0.03 / 3.6

                frame_info['distance'] = dist

            if frame_info['xtl'] < 0 or frame_info['xbr'] > image_res[1] or frame_info['ytl'] < 0 or frame_info['ybr'] > image_res[0]:
                out_data.get(id)['critical_point'] = frame_index - 1
                break

            frame_info_list.append(frame_info)

        if not frame_info_list:
            del out_data[id]
            print('deleted bacause of no visualize data')
            continue
        else:
            out_data.get(id)['frames_info'] = frame_info_list
            rep_id_list = [id]

            for rep in range(2, int((out_data.get(id).get('critical_point') - out_data.get(id).get('start_point')) // 0.5)):
                clip_length = rep * 0.5
                rep_id = f"{id}_{clip_length}"
                out_data[rep_id] = out_data.get(id)
                out_data.get(rep_id).get('start_point') = out_data.get(rep_id).get('critical_point') - int(clip_length * 30)
                out_data.get(rep_id).get('frames_info') = out_data.get(rep_id).get('frames_info')[int(clip_length * 30):]
                rep_id_list.append(rep_id)

        for rep_id in rep_id_list:

            if os.path.isfile(f'{out_file}/clips/{rep_id}.mp4'):
            # if label == 'pedestrian' and os.path.isfile(f'{out_file}/clips/{id}.mp4'):
                print('skiped pedestrian clip bacause the file exists')
                continue

            print(f"start get video frame from {out_data.get(rep_id).get('start_point')} to {out_data.get(rep_id).get('critical_point')}")

            # frame_list = []
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
            writer = cv2.VideoWriter(f'{out_file}/clips/{rep_id}.mp4', fourcc, frame_rate, (image_res[1], image_res[0]))
            for index in range(out_data.get(rep_id).get('start_point'), out_data.get(rep_id).get('critical_point')):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = video.read()
                if ret:
                    frame = cv2.resize(frame[crop_value[0]:crop_value[1], crop_value[2]:crop_value[3]], dsize=None, fx=expand_rate, fy=expand_rate)
                    writer.write(frame)
                    # frame_list.append(frame)
                else:
                    break

            writer.release()
        # out_data.get(rep_id)['frames'] = frame_list



def main(args):

    out_data = {}
    i=0
    for video_file in glob.iglob(args.video_dir+'/set*/*.mp4'):
        i+=1
        set_name = video_file.split('/')[-2]
        video_name = video_file.split('/')[-1].split('.')[-2]
        print('start', video_name)
        attrib_file = args.attrib_dir+'/'+set_name+'/'+video_name+'_attributes.xml'
        annt_file = args.annotation_dir+'/'+set_name+'/'+video_name+'_annt.xml'
        vehicle_file = args.vehicle_dir+'/'+set_name+'/'+video_name+'_obd.xml'

        if os.path.isfile(attrib_file) and os.path.isfile(annt_file):

            video, _, _, crop_value = PieLib.getVideo(video_file, args.image_offset_y, args.crop_rate)
            attrib = PieLib.getXmlRoot(attrib_file)
            annt = PieLib.getXmlRoot(annt_file)
            vehicle = PieLib.getXmlRoot(vehicle_file)

            pieExtractClip(video, crop_value, args.crop_rate, attrib, annt, vehicle, out_data, args.out_file)
        print('done')

    with open(f'{args.out_file}/data.pickle', mode='wb') as file:
        pickle.dump(out_data, file)
    print('saved_pickle')

    # if i==3: return

if __name__ == '__main__':

    argparser = argparse.ArgumentParser( description = __doc__)
    argparser.add_argument(
        '--video_dir',
        default='/media/ssd/PIE_data/PIE_clips',
        metavar='DIR')
    argparser.add_argument(
        '--annotation_dir',
        default='/media/ssd/PIE_data/annotations',
        metavar='DIR')
    argparser.add_argument(
        '--attrib_dir',
        default='/media/ssd/PIE_data/annotations_attributes',
        metavar='DIR')
    argparser.add_argument(
        '--vehicle_dir',
        default='/media/ssd/PIE_data/annotations_vehicle',
        metavar='DIR')

    argparser.add_argument(
        '--crop_rate',
        metavar='SCALE',
        default=0.6)
    argparser.add_argument(
        '--image_offset_y',
        metavar='OFFSET',
        default=0.2)
    argparser.add_argument(
        '--out_file',
        metavar='DIR',
        default='/media/ssd/PIE_data/extracted_data')

    args = argparser.parse_args()

    main(args)
