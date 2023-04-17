import json
import numpy as np
import torch
import heapq

alldata = {}
with open('/Users/graceli/Desktop/result/GMP_1960_test0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

with open('/Users/graceli/Desktop/result/GMP_1960_test1.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
   
with open('/Users/graceli/Desktop/result/GMP_1960_test2.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_test3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_test4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_test5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_test6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_test7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)

testdata = alldata
alldata = {}
with open('/Users/graceli/Desktop/result/GMP_1960_valid0.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid1(1).json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid2(3).json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid3.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid4.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid5.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid6.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
with open('/Users/graceli/Desktop/result/GMP_1960_valid7.json') as json_file:
   data = json.load(json_file)
   alldata.update(data)
   
validdata = alldata
   
   
data = validdata

   
  
output_data = []
for music in data:
    output = data[music]["output"]
    if len(output) > 4:
        heapq.heapify(output)
        n = int(len(output) * 1)
        n_largest = heapq.nlargest(n, output)
        
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
print(output_data)
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
