#!/usr/bin/env python
# -*- coding:utf-8 -*-

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw_roc(ax):

    d_list = [0.0, 0.5, 1.0, 2.0, 3.0]
    for d in d_list:
        C = np.linspace(-10, 10+d, 101)
        FA = [norm.cdf(x) for x in C]
        hit = 1.0-np.array([norm.cdf(-x-d) for x in C])
        sns.lineplot(x=FA, y=hit, ax=ax)

    # plt.show()
    return ax


def calc_d_C(hit, FA):
    d = norm.ppf(hit) - norm.ppf(FA)
    C = -0.5 * (norm.ppf(hit) + norm.ppf(FA))
    d = np.nan if d == np.inf else d
    C = np.nan if C == np.inf else C
    return d, C


def draw_data_on_roc(data, ax):
    db = {} # {subject: {hit:[0.0, 0.5, 1.0], FA:[0.0, 0.5, 1.0]}}
    for subject in data.subject.drop_duplicates():
        target_data = data[data.subject == subject]
        hit = len(target_data[target_data.acc_pred==0])/len(target_data)
        FA = len(target_data[target_data.acc_pred==2])/len(target_data)
        sns.lineplot(x=FA, y=hit, ax=ax, label=subject)

        # buf = {
        #     "hit": len(target_data[target_data.acc_pred==0])/len(target_data),
        #     "FA": len(target_data[target_data.acc_pred==2])/len(target_data),
        #     }
        # db[subject] = buf


data = pd.read_csv("/home/kuriatsu/Documents/experiment/pie_202201/summary.csv")
fig, ax = plt.subplots()
ax = draw_roc(ax)

db_d_C = pd.DataFrame(columns=["subject", "d", "C", "thresh"]) # {subject: [d,C]}
for subject in data.subject.drop_duplicates():
    hit_list = []
    FA_list = []
    for thresh in data.recognition_thresh.drop_duplicates():
        target_data = data[(data.subject == subject) & (data.recognition_thresh == thresh)]
        hit = len(target_data[target_data.responce_int_vs_prob==0])/len(target_data)
        FA = len(target_data[target_data.responce_int_vs_prob==2])/len(target_data)
        hit_list.append(hit)
        FA_list.append(FA)
        d, C = calc_d_C(hit, FA)
        buf = pd.DataFrame(np.array([subject, d, C, thresh]).reshape(1, -1), columns=db_d_C.columns)
        db_d_C = pd.concat([db_d_C, buf])
    sns.lineplot(x=FA_list, y=hit_list, ax=ax, label=subject)


plt.show()
