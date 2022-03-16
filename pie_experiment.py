#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
import random
from pie_ras import PIERas

def main():
    playlist = []
    set_num = 2
    thres = 0.0
    subject = "suzuki"
    with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/playlist_{}.csv".format(subject), "r") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader]
    random.shuffle(playlist[set_num])

    with open("/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/database_result_valid.pkl", 'rb') as f:
        database = pickle.load(f)

    with PIERas() as pie_visualize:
        for id in playlist[set_num]:
            try:
                pie_visualize.log_file = "/media/kuriatsu/SamsungKURI/PIE_data/extracted_data/log_data_{}_{}.csv".format(subject, set_num)
                pie_visualize.is_checked_thres = thres
                pie_visualize.play(database.get(id))
            except KeyboardInterrupt:
                break
            except:
                pass

if __name__ == "__main__":
    main()
