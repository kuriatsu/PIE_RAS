#! /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import argparse
import numpy as np
import time
import datetime
import csv
import pickle
import random



data_file = "/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/database.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)
