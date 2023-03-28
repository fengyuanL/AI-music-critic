import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import requests
from requests.exceptions import ReadTimeout
import json

    
with open('/Users/graceli/Desktop/ECE324/Project/popularityDict.txt') as json_file:
    dataWithPop = json.load(json_file)   # 7231 music pieces in total


with open('/Users/graceli/Desktop/ECE324/Project/DataDictFull.json') as json_file:
    data = json.load(json_file) 

dictwithPop = {}
dictnoPop = {}

for music in data:
    if music in dataWithPop:
        dictwithPop[music] = data[music]
        data[music]["Popularity"] = dataWithPop[music]["Popularity"]
    else:
        dictnoPop[music] = data[music]

with open('/Users/graceli/Desktop/ECE324/Project/dictnoPop.json', 'w') as convert_file1:
  convert_file1.write(json.dumps(dictnoPop))
  
with open('/Users/graceli/Desktop/ECE324/Project/dictwithPop.json', 'w') as convert_file2:
  convert_file2.write(json.dumps(dictwithPop))
  
# with open("/Users/graceli/Desktop/ECE324/Project/dictnoPop.txt", "w") as f:
#     f.write(json.dumps(dictnoPop))
    
# with open("/Users/graceli/Desktop/ECE324/Project/dictwithPop.txt", "w") as f:
#     f.write(json.dumps(dictwithPop))

print(len(dictnoPop))
print(len(dictwithPop))
