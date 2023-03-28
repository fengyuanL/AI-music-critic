import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import requests
from requests.exceptions import ReadTimeout
import json

    
# with open('/Users/graceli/Desktop/ECE324/Project/GetPopularity/DataDictFull_final.json') as json_file:
#     data_final = json.load(json_file)   # 7231 music pieces in total


with open('/Users/graceli/Desktop/ECE324/Project/GetPopularity/DataDictFullwithPop3_20.json') as json_file:
    data = json.load(json_file) 

count = 0

for music in data:
    
    if data[music]["Composition Date"] == "":
        count += 1

print(count)

# with open('/Users/graceli/Desktop/ECE324/Project/GetPopularity/DataDictFullwithPop3_20.json', 'w') as convert_file1:
#   convert_file1.write(json.dumps(data))
  
# with open("/Users/graceli/Desktop/ECE324/Project/dictnoPop.txt", "w") as f:
#     f.write(json.dumps(dictnoPop))
    
# with open("/Users/graceli/Desktop/ECE324/Project/dictwithPop.txt", "w") as f:
#     f.write(json.dumps(dictwithPop))


