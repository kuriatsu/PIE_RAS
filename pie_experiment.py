#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
from pie_visualize import PIEVisualize

def main():
    pie_visualize = PIEVisualize()

    with open("/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/playlist.csv", "rb") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader][sys.argv[1]]

    with open("/home/kuriatsu/media/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
        database = pickle.load(f)

    for id in playlist:
        pie_visualize.play(database.get(id))

if __name__ == "__main__":
    main()
