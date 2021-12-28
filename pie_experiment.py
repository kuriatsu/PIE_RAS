#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
import random
from pie_visualize import PIEVisualize

def main():
    pie_visualize = PIEVisualize()
    playlist = []
    with open("/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist.csv", "r") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader]

    with open("/run/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
        database = pickle.load(f)

    random.shuffle(playlist[int(sys.argv[1])])
    for id in playlist[int(sys.argv[1])]:
        pie_visualize.play(database.get(id))

if __name__ == "__main__":
    main()
