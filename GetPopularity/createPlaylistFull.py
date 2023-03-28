import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import requests
from requests.exceptions import ReadTimeout
import json


def check_similar(ph1, ph2):
  # return ph1 == ph2
  # ph1 is the original sentence, ph2 is the given to verify
  words1 = (ph1.lower()).split()
  ph2 = ph2.lower()
  count = 0 
  for word in words1:
    if ph2.find(word) == -1:
      count += 1
    #   print(word)
  if count <= 0:
    return True
  else:
    return False

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# ######
# from google.colab import drive
# drive.mount('/content/drive')
##Change these:
username = 'cr53p7y6ag4m4ykszraxsbcaz'  # your username (not an email address)

createnewplaylist = True  ## Set this to true to create a new playlist with the name below; set this to false to use an already created playlist, and follow instructions below
newplaylistname = 'dataset1'     ############ adjust this


# dataFile = "/content/drive/MyDrive/ECE324/spotifysearch.txt"
# dataFile = "/Users/graceli/Desktop/ECE324/Project/dictnoPop.txt"


    
with open('/Users/graceli/Desktop/ECE324/Project/dictnoPop.json') as json_file:
    data = json.load(json_file)   # 7231 music pieces in total


my_client_id = '878b93284f1c4d24801a92979995bfaa'
my_client_secret = 'b5914c9a22f242868867257baf895a57'
######
######

count = 0

scope = 'user-library-read playlist-modify-public playlist-modify-private'

# data = open(dataFile).readlines()[0:500]    ############## adjust this

# with open('/Users/graceli/Desktop/ECE324/Project/dictnoPop.txt') as json_file:
#     data = json.load(json_file) 

# token = util.prompt_for_user_token(username,scope,client_id=my_client_id,client_secret=my_client_secret,redirect_uri='http://localhost/') 
# sp = spotipy.Spotify(auth=token)

token = util.prompt_for_user_token(username, scope, client_id=my_client_id, client_secret=my_client_secret,
                                   redirect_uri='http://localhost:3333/callback')
                                #    redirect_uri='http://localhost:8888/callback')

myAuth = "Bearer " + token
# myAuth = "Bearer BQBdW2prx7jp5KFXUDOScSMSnLrCWa0DV9Rx_IXVoOfzx_ros8gtKoJCXzUngj_oeaJEFIEHr-hh5GMs9-9ZSoQjrwiEmrlgbH35MS-bDazYzYxioP7WTSKMn4teBvrvnAji5_WOxeXj4q7cvEqCo4F3HmrK3NDL4igSlQutN_vySu5jQraCj2j0GkVdu34oxPZybgBw-GJHLGiJGKNk6xCgiawy_VLZz2LnaShsNCtueLanXKdl2YcDphViTR-O"

notfound = []
count_notfound = 0
output = {}
# input_data = data[0:500]

if token:
    sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)

    if createnewplaylist:
        r = sp.user_playlist_create(username, newplaylistname, False)
        playlistID = r['id']
    # else:
    #     playlistID = oldplaylistID

for dictname in data:        ######## change here
    count += 1
    # get a new token
    if count == 100:
        with open("/Users/graceli/Desktop/ECE324/Project/DictwithPopularity.txt", "w") as f:
            f.write(json.dumps(output))
        token = util.prompt_for_user_token(username, scope, client_id=my_client_id, client_secret=my_client_secret,
                                   redirect_uri='http://localhost:3333/callback')
                                #    redirect_uri='http://localhost:8888/callback')

        myAuth = "Bearer " + token
        sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)
        count = 0


    l = dictname.split(",")
    trackTitle = " ".join(l[2:])
    artist = l[1] + " " + l[0] 

        
    r = 0
        # r = sp.search(q="artist:" + artist + " track:" + trackTitle, type="track")
    try:
        r = sp.search(q=artist + trackTitle, type="track,album")
    except ReadTimeout:
        print('Spotify timed out... trying again...')
        sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)
        r = sp.search(q=artist + trackTitle, type="track,album")


    found = False
    for track in r['tracks']['items']:
        if check_similar(artist.lower(), track['artists'][0]['name'].lower()):
            trackID = track['id']
            found = True
            track_info = sp.track(trackID)
            data[dictname]["Popularity"] = track_info["popularity"]
            output[dictname] = data[dictname]
            print("\"", dictname,"\" : ", data[dictname],",")
            break

    if not found:
            # print '****  Could not find song',trackTitle,'by artist',artist
        data[dictname]["Popularity"] = 0
        output[dictname] = data[dictname]   ###
        print("\"", dictname,"\" : ", data[dictname],",")
        count_notfound += 1
        notfound.append(trackTitle + "  " + artist)
            
    else:
        requests.post(
            "https://api.spotify.com/v1/users/" + username + "/playlists/" + playlistID + "/tracks?position=0&uris=spotify%3Atrack%3A" + trackID,
            headers={"Authorization": myAuth})
        print('Added song', trackTitle, 'by artist', artist)


with open("/Users/graceli/Desktop/ECE324/Project/DictwithPopularity.txt", "w") as f:
    f.write(json.dumps(data))
    
print("Count find popularity of :", count_notfound)

##??????????
#   "Reichert, Matheus Andre\u0301, Tarentelle, Op.3": {"File_Name": "Reichert, Matheus Andre\u0301, Tarentelle, Op.3, XRqcO-OtepI.mid", 
# "Surname": "Reichert", "First_Name": "Matheus Andre\u0301", "Piece_Name": "Tarentelle, Op.3", "Similarity": "XRqcO-OtepI", 
# "url": "https://imslp.org/wiki/Tarentelle%2C_Op.3_%28Reichert%2C_Matheus%20Andre%CC%81%29", "Composition Date": ""},

# if check_similar(artist.lower(), track['artists'][0]['name'].lower()):
# TypeError: 'NoneType' object is not subscriptable