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


def main():
    with open("/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
        pie_experiment.database = pickle.load(f)
        pie_experiment.loop()


if __name__ == "__main__":
    main()
