#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
import random
from pie_ras import PIERas

def main():
    playlist = []
    with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist.csv", "r") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader]
    random.shuffle(playlist[int(sys.argv[1])])

    with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database.pkl", 'rb') as f:
        database = pickle.load(f)

    with PIERas() as pie_visualize:
        for id in playlist[int(sys.argv[1])]:
            try:
                pie_visualize.log_file = "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/log_data_{}_kanayama.csv".format(sys.argv[1])
                pie_visualize.play(database.get(id))
            except KeyboardInterrupt:
                break
            except:
                pass

if __name__ == "__main__":
    main()
