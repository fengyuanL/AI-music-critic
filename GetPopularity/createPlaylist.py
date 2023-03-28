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
newplaylistname = 'ECE324 dataset19'     ############ adjust this
## If using an already existing playlist, go to Spotify and right click on a playlist and select "Copy Spotify URI". Paste the value below, keeping only the numbers at the end of the URI
# oldplaylistID='3uEcg6o2uf2ijoyeRj3zLiF'

# dataFile = "/content/drive/MyDrive/ECE324/spotifysearch.txt"
dataFile = "/Users/graceli/Desktop/ECE324/Project/spotifysearch.txt"

with open('/Users/graceli/Desktop/ECE324/Project/DataDict.json') as json_file:
    dataDict = json.load(json_file)   # 7231 music pieces in total

delim = "\t"  # charecters between song title and artist in your data file; make sure this is not something that could be present in the song title or artist name

my_client_id = '878b93284f1c4d24801a92979995bfaa'
my_client_secret = 'b5914c9a22f242868867257baf895a57'
######
######

count = 0

scope = 'user-library-read playlist-modify-public playlist-modify-private'

data = open(dataFile).readlines()[7000:]    ############## adjust this

# token = util.prompt_for_user_token(username,scope,client_id=my_client_id,client_secret=my_client_secret,redirect_uri='http://localhost/') 
# sp = spotipy.Spotify(auth=token)

token = util.prompt_for_user_token(username, scope, client_id=my_client_id, client_secret=my_client_secret,
                                   redirect_uri='http://localhost:3333/callback')
                                #    redirect_uri='http://localhost:8888/callback')

myAuth = "Bearer " + token
# myAuth = "Bearer BQBdW2prx7jp5KFXUDOScSMSnLrCWa0DV9Rx_IXVoOfzx_ros8gtKoJCXzUngj_oeaJEFIEHr-hh5GMs9-9ZSoQjrwiEmrlgbH35MS-bDazYzYxioP7WTSKMn4teBvrvnAji5_WOxeXj4q7cvEqCo4F3HmrK3NDL4igSlQutN_vySu5jQraCj2j0GkVdu34oxPZybgBw-GJHLGiJGKNk6xCgiawy_VLZz2LnaShsNCtueLanXKdl2YcDphViTR-O"

notfound = []
count_notfound = 0

if token:
    sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)

    if createnewplaylist:
        r = sp.user_playlist_create(username, newplaylistname, False)
        playlistID = r['id']
    # else:
    #     playlistID = oldplaylistID

    for line in data:
        count += 1

        l = line.split(delim)
        dictname = l[0] + ", " + l[1] + ", " + l[2][:-1] 
        trackTitle = l[    
            -1]  ## If you have any characters after your track title before your delimiter, add [:-1] (where 1 is equal to the number of additional characters)
        artist = l[1] + " " + l[
            0]  ## [:-1] removes the newline at the end of every line. Make this [:-2] if you also have a space at the end of each line

        r = 0
        # r = sp.search(q="artist:" + artist + " track:" + trackTitle, type="track")
        try:
            r = sp.search(q=artist + trackTitle, type="track,album")
        except ReadTimeout:
            print('Spotify timed out... trying again...')
            sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=10)
            r = sp.search(q=artist + trackTitle, type="track,album")


        found = False
        # if count == 5:
        # print(r)
        # print(r['tracks']['items'])
        for track in r['tracks']['items']:
            # print(track['artists'][0]['name'].lower(), artist.lower())
            if check_similar(artist.lower(), track['artists'][0]['name'].lower()):
                trackID = track['id']
                found = True
                track_info = sp.track(trackID)
                dataDict[dictname]["Popularity"] = track_info["popularity"]
                print("\"", dictname,"\" : ", dataDict[dictname],",")
                break

        if not found:
            # print '****  Could not find song',trackTitle,'by artist',artist
            dataDict[dictname]["Popularity"] = 0
            print("\"", dictname,"\" : ", dataDict[dictname],",")
            count_notfound += 1
            notfound.append(trackTitle + delim + artist)
            
        else:
            requests.post(
                "https://api.spotify.com/v1/users/" + username + "/playlists/" + playlistID + "/tracks?position=0&uris=spotify%3Atrack%3A" + trackID,
                headers={"Authorization": myAuth})
            print('Added song', trackTitle, 'by artist', artist)

    print(count)

    # print("\nSongs not added: ")
    # for line in notfound:
    #     print(line)
    # print("\n")

else:
    print("Can't get token for", username)

with open("/Users/graceli/Desktop/ECE324/Project/DictwithPopularity.txt", "w") as f:
#   for item in dataDict:
    f.write(json.dumps(dataDict))
    
print("Count find popularity of :", count_notfound)