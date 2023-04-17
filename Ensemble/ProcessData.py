import numpy as np
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import math
import random
import pickle
from tqdm import tqdm

seq = np.load('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_reproc_train.txt.npy')
a = seq[0][0][0][0]
# print(a)
# print(len(a))

distribution = np.histogram(a, bins=30, range=(-3,3), density=True, weights=None)[0]
print(distribution)