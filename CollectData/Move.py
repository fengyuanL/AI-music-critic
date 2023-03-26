import os
import json
import matplotlib.pyplot as plt
filename = 'DataDictFull_time.json'
file_handle = open(filename)
date_detail = json.loads(file_handle.read())
import shutil
midi_path = 'D:/UT/ECE324/MIDI-BERT/Data/Dataset/GMP'

DataDict = {}
pieces = []
error_count = 0

for root, dirs, files in os.walk(midi_path):
    for count, filename in enumerate(files):
        KEY = filename[:filename.rfind(", ")]
        if KEY not in date_detail:
            print(KEY, "NOT found")