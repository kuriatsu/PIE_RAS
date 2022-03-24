#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
import random
from pie_ras import PIERas

def main():
    subject = "kuribayashi"
    trial = 0
    type = "tl"

    playlist = []
    type_code = {"int":0, "traj":1, "tl":2}
    set_num = 3 * trial + type_code.get(type)
    with open("./data/PIE_data/experiment/playlist.csv".format(subject), "r") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader]
    random.shuffle(playlist[set_num])

    with open("./data/PIE_data/experiment/database.pkl", 'rb') as f:
        database = pickle.load(f)

    with PIERas(type) as pie_visualize:
        for id in playlist[set_num]:
            try:
                pie_visualize.log_file = "./data/PIE_data/experiment/log_data_{}_{}.csv".format(subject, set_num)
                pie_visualize.play(database.get(id), "result")
            except KeyboardInterrupt:
                break
            except:
                pass

if __name__ == "__main__":
    main()
