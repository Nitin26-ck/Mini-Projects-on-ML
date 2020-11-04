# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:14:46 2020

@author: Nitin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('E:/Machine Learning/P14-Part6-Reinforcement-Learning/P14-Part6-Reinforcement-Learning/Section 31 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv')

import random
N=10000
d=10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward    

plt.hist(ads_selected)
plt.title('histogram of ads selected')
plt.xlabel('ads')
plt.ylabel('no of times each ad was selected')
plt.show()