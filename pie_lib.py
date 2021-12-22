#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import xml.etree.ElementTree as ET
import glob

class PieLib():

    def getVideo(filename, image_offset_y, crop_rate):

        try:
            video = cv2.VideoCapture(filename)

        except:
            print('cannot open video')
            exit(0)

        # get video rate and change variable unit from time to frame num
        fps = int(video.get(cv2.CAP_PROP_FPS))
        image_res = [video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)]
        # adjust video rate to keep genuine broadcast rate

        # calc image-crop-region crop -> expaned to original frame geometry
        offset_yt = image_res[0] * ((1.0 - crop_rate) * 0.5 + image_offset_y)
        offset_xl = image_res[1] * (1.0 - crop_rate) * 0.5
        crop_value = [int(offset_yt),
                      int(offset_yt + image_res[0] * crop_rate),
                      int(offset_xl),
                      int(offset_xl + image_res[1] * crop_rate)
                      ]

        return video, image_res, fps, crop_value


    def getXmlRoot(filename):

        # try:
        tree = ET.parse(filename)
        return tree.getroot()
        del tree
        # except:
        #     print(f'cannot open {filename} file')
        #     exit(0)


    def prepareIcon(filename):
        icon_dict = {}

        for icon_file in glob.iglob(filename+'/*.png'):
            img = cv2.imread(icon_file)
            name = icon_file.split('/')[-1].split('.')[-2]
            roi = img.shape[:2]
            img2grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2grey, 10, 255, cv2.THRESH_BINARY)
            # ret, mask = cv2.threshold(img2grey, 200, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            icon_fg = cv2.bitwise_and(img, img, mask=mask)

            icon_dict[name] = {'roi': roi, 'mask_inv': mask_inv, 'icon_fg': icon_fg}

        return icon_dict


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
