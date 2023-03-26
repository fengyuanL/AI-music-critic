import os
import json
import matplotlib.pyplot as plt
filename = 'DataDictFull_time.json'
file_handle = open(filename)
date_detail = json.loads(file_handle.read())
print("Total Size: ", len(date_detail.keys()))

counter = 0
empty = 0
finished = 0
year_lst = []
for dict_key in date_detail.keys():
    row = date_detail[dict_key]
    if 'Composition Date' in row:
        finished += 1
        if row['Composition Date'] == '':
            empty += 1
            # print(dict_key)
        else:
            year_lst.append(int(row['Composition Date']))
        continue

print(empty)
print(finished)

plt.hist(year_lst, bins=100)
plt.show()

date_detail['Carafa, Michele, La violette']['Composition Date'] = 1828
date_detail['Cappa, Antonio José, Recuerdos de la Alhambra']['Composition Date'] = 1890
1889