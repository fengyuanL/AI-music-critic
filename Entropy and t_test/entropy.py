import torch
import numpy as np
from scipy.stats import entropy
import time
import heapq
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# seq = np.load('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_reproc_train.txt.npy')
# hs = seq[0][0]    # 1, 512, 768

alldata = {}

###### 2023
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_test7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)


with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_2023_valid7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)


###### 1960
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_test7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)


with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1960_valid7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
###### 1860
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_test7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)


with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/MIDI-BERTFT/Data/CP_data/GMP_1860_valid7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   

validdata = alldata
   
   
data = validdata

all_data = []
for music in data:
    entropy_list = data[music]["Entropy"]
    piece_entropy = 0
    for sequence in entropy_list:
      avg = np.mean(sequence[4:12])
      piece_entropy += avg
      
    total_avg = piece_entropy
    total_avg = total_avg/len(entropy_list)
    if total_avg > 50:
       print(total_avg)
    pop = int(data[music]["pop"])
    if pop != 0:
   #      for i in range(0, 4): 	
      all_data.append([total_avg, pop])
   #  else:
   #  all_data.append([total_avg, pop])

# all_data = np.array(all_data)
# all_entropy = all_data[:,0]
# all_pop = all_data[:,1]

# all_data = np.array(all_data)
# all_entropy = all_data[:,0]
# all_pop = all_data[:,1]

# sorted(all_data,key=lambda l:l[0], reverse=True)
# print(list(all_data))
all_data = np.array(all_data)
all_entropy = all_data[:,0]
# all_pop = all_data[:,1]
all_pop = np.emath.logn(1.5157, all_data[:,1] + 1)

print(list(all_pop))
# print(list(all_entropy))

valid_idx = np.isfinite(all_entropy)
all_entropy = all_entropy[valid_idx]
all_pop = all_pop[valid_idx]
bin_means, bin_edges, binnumber = stats.binned_statistic(all_entropy, all_pop, statistic='mean', bins=10, range=[1.5,2])
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

print(bin_means)
print(bin_centers)
# plt.scatter(all_entropy, all_pop, s=1)
# plt.xlabel("Entropy")
# plt.ylabel("Popularity")
# plt.show()

plt.plot(bin_centers,bin_means)
# plt.hist2d(all_entropy, all_pop, bins=(100, 100), cmap=plt.cm.jet, range=[[1.5,2],[0,10]])

plt.xlabel("Entropy")
plt.ylabel("Popularity")
plt.show()




# def calculate_entropy(hs):
#     # 512 vectors
#     # a vector has 768 elements
#     # Get entropy of each vector

#     all_entropy = []
#     heapq.heapify(all_entropy)
#     for neuron in range(768):
#         vector = hs[0][:, neuron]
#         distribution = np.histogram(vector, bins=6, range=(-3,3), density=True, weights=None)[0] 
#         H = entropy(distribution,base=2)
#         heapq.heappush(all_entropy, H)
    
#     n_largest = heapq.nlargest(614, all_entropy)
    
#     return sum(n_largest)/614

# print(calculate_entropy(hs))

# start_time = time.time()
# print(calculate_entropy(hs))
# end_time = time.time()
# time2 = end_time - start_time
# print("time1: ", time2)

# def calculate_entropy(hs):
#     # 512 vectors
#     # a vector has 768 elements
#     # Get entropy of each vector

#     all_entropy = []
#     heapq.heapify(all_entropy)
#     for neuron in range(768):
#         vector = hs[0][:, neuron]
#         distribution = np.histogram(vector, bins=6, range=(-3,3), density=True, weights=None)[0] 
#         H = entropy(distribution,base=2)
#         heapq.heappush(all_entropy, H)
    
#     n_largest = heapq.nlargest(614, all_entropy)
    
#     return sum(n_largest)/614

# def calculate_entropy(hs):
#     print("length of vectors: ", len(hs[0]))
#     total_H = 0
#     for vector in hs[0]:
#         unique, counts = np.unique(vector, return_counts=True)
#         frequency = np.asarray((unique, counts))
#         # print("frequency:", frequency)
#         pk = frequency[1]/sum(frequency[1])
#         # print("pk:", pk)
        
#         H = -np.sum(pk * np.log(pk)) / np.log(2)
#         # print(H)
#         total_H += H
#     return total_H

# # start_time = time.time()
# # print(calculate_entropy(hs))
# # end_time = time.time()
# # time2 = end_time - start_time
# # print("time2: ", time2)


