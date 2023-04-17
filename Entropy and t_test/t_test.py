import torch
import numpy as np
from scipy import stats 
from numpy.random import seed
from numpy.random import randn
from numpy.random import normal
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

from queue import PriorityQueue
import json

# seed the random number generator
seed(1)
# create two independent sample groups
sample1= [10.2796, 0.6843, 10.2299, 0.6994]
sample2= [8.1514, 0.7665, 7.8307, 0.7747]


t_stat, p_value = ttest_ind(sample1, sample2)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)