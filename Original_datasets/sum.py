import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from itertools import compress

name = 'toxcast'

f = open(os.path.join(BASE_DIR, '{}/{}_data.csv'.format(name,name)), 'r').readlines()[1:]
np.random.shuffle(f)


lable_list = []
for i in range(617):
    lable_list.append([0,0])

if __name__ == "__main__":
    num = 0
    for index, line in enumerate(f):
        l = line.split(",")
        for i in range(618):
            if i != 0:
                if l[i] == "0.0":
                    lable_list[i-1][0] += 1
                elif l[i] == "1.0":
                    lable_list[i-1][1] += 1
            if i == 617:
                if l[i][:-1] == "0.0":
                    lable_list[i-1][0] += 1
                elif l[i][:-1] == "1.0":
                    lable_list[i-1][1] += 1
        num += 1

    print(lable_list)