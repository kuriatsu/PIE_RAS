#! /usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import pickle
import sys
import random
from pie_ras import PIERas

def main():
    subject = "test"
    trial = 0
    # trial = 21
    # type = "traj"
    # type = "int"
    type = "tl"

    playlist = []
    type_code = {"int":0, "traj":1, "tl":2}
    set_num = 3 * trial + type_code.get(type)
    # set_num = 0
    # with open("/home/kuriatsu/Dropbox/data/pie202203/false_list.csv".format(subject), "r") as f:
    with open("./data/PIE_data/experiment/playlist.csv".format(subject), "r") as f:
        reader = csv.reader(f)
        playlist = [row for row in reader]
    random.shuffle(playlist[set_num])
    print(playlist)

    with open("./data/PIE_data/experiment/database.pkl", 'rb') as f:
        database = pickle.load(f)

    with PIERas(type) as pie_visualize:
        for id in playlist[set_num]:
            try:
                pie_visualize.log_file = "./data/PIE_data/experiment/log_data_{}_{}_{}.csv".format(subject, type, set_num)
                pie_visualize.play(database.get(id), "result")
            except KeyboardInterrupt:
                break
            except:
                pass

if __name__ == "__main__":
    main()
