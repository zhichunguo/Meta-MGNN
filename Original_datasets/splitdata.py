import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from itertools import compress

import random

import json

name = 'tox21'


f = open(os.path.join(BASE_DIR, '{}/raw/{}.csv'.format(name,name)), 'r').readlines()[1:]
np.random.shuffle(f)


if __name__ == "__main__":
    tasks = {}
    
    # Below needs to be modified according to different original datasets
    for index, line in enumerate(f):
        l = line.split(",")
        # print(l[27][:-1])
        for i in range(618):
            if i != 0:
                if i not in tasks:
                    tasks[i] = [[],[]]
                if i == 617:
                    l[i] = l[i][:-1]
                if l[i] == "0.0":
                    tasks[i][0].append(l[0])
                elif l[i] == "1.0":
                    tasks[i][1].append(l[0])
    #until here

    for i in tasks:
        root = name + "/new/" + str(i)
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + "/raw", exist_ok=True)
        os.makedirs(root + "/processed", exist_ok=True)

        file = open(root + "/raw/" + name + ".json", "w")
        file.write(json.dumps(tasks[i]))
        file.close()
    
