# Code from:
# https://www.javatpoint.com/spotify-api-python#:~:text=One%20of%20the%20advantages%20of,creating%20and%20managing%20Spotify%20apps.

import spotipy  
from spotipy.oauth2 import SpotifyOAuth  
import spotipy 
from spotipy import Spotify  
  
sp_oauth = SpotifyOAuth(client_id="c667d9211520403ca4151abd75314625", client_secret="b666c32cb0924b97809ca92d6dce4178", redirect_uri="http://localhost/")  
  
access_token = sp_oauth.get_access_token()  
# refresh_token = sp_oauth.get_refresh_token()  
  
sp = Spotify(auth_manager = sp_oauth)

results = sp.search(q='track:dancing', type='track')

print(results)