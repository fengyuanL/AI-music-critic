import torch
import numpy as np
from scipy import stats 
from numpy.random import seed
from numpy.random import randn
from numpy.random import normal
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

from sklearn.metrics import mean_squared_error


import json
import numpy as np
import torch
import heapq

alldata = {}
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_test7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)


with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_Post1910_valid7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
validdata = alldata
   
   
data = validdata

   
  
output_data = []
for music in data:
    output = data[music]["output"]
    if len(output) > 4:
        heapq.heapify(output)
        n = int(len(output))
        n_largest = heapq.nlargest(n, output)
    
    else:
        n_largest = output
    a = int(data[music]["pop"])
    pop = np.emath.logn(1.5157, a + 1)
    avg_output = sum(n_largest)/len(n_largest)
    
    # output_data[music] = {}
    # # print(tensor)
    # output_data[music]["average output"] = avg_output
    # output_data[music]["pop"] = pop
    if pop != 0:
        for i in range(0, 4): 	
            output_data.append([avg_output,pop ])
    else:
        output_data.append([avg_output, pop])
  


torch.tensor(output_data)
output = torch.tensor(output_data)[:, 0]
y = torch.tensor(output_data)[:, 1]
threshold = 4


def accuracy(t1, t2):
    if len(t1) != len(t2):
        print("Warning: Not equal length in accuracy")
    if len(t1) > len(t2):
        t1 = t1[:len(t2)]
    elif len(t2) > len(t1):
        t2 = t2[:len(t1)]
    return torch.tensor(sum(torch.logical_and(t1, t2)) / len(t1), dtype=float)


threshold = 4
accl = accuracy(y <= threshold, output <= threshold) # accuracy of not acclaimed from whole sequence
accr = accuracy(y > threshold, output > threshold) # accuracy of acclaimed from whole sequence
acc = accl + accr
print(acc, accl, accr)

# MSELoss
print(mean_squared_error(y,output))
MSE = np.square(np.subtract(y,output)).mean()
print(MSE)

# # seed the random number generator
# seed(1)
# # create two independent sample groups
# sample1= normal(30, 16, 50)
# sample2=normal(33, 18, 50)
# print('Sample 1: ',sample1)
# print('Sample 2: ',sample2)


# t_stat, p_value = ttest_ind(sample1, sample2)
# print("T-statistic value: ", t_stat)  
# print("P-Value: ", p_value)