import numpy as np
import os
import re
import csv

class CALIBDATA:
    def __init__(self):
        K1 = []
        K2 = []

def load_data(data_path):
    data = {}

    dataLoad = os.listdir(data_path)
    dataLoad.sort()

    for index, file in enumerate(dataLoad):
        if (index % 3) == 0:
            key = re.split(r'_|\.', file)[0][:-1]
            data[key] = []
        
        data[key].append(file)
    
    return data

def read_calib(root, calib_path):
    calibData = CALIBDATA()

    with open(root + calib_path, 'r') as file:
        calib_file = file.readlines()
    data_reader = csv.reader(calib_file, delimiter=' ')

    data = []
    indexArr = [3, 4, 5, 9, 10, 11]
    for index, row in enumerate(data_reader):
        if index in indexArr:
            row = [float(row[i]) for i in range(len(row))]
            data.append(row)
    file.close()

    calibData.K1 = np.vstack((data[0], data[1], data[2]))
    calibData.K2 = np.vstack((data[3], data[4], data[5]))
    
    return calibData