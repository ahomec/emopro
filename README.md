# Emotion-based music player
This project utilizes human facial recognition to identify the emotions of the user and then recommend them a song that matches using a music emotion classification algorithm trained with Spotify data gathered from the API.

We utilize a CNN neural network to build the facial emotion recognition model and k-nearest neighbors classification to build the music emotion classification model.


### Spotify Music Classification Model

### Data

The dataset used to train the model was found here:

Spotify provides metrics for each song that we used to classify emotion.

* Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
* Danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
* Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
* Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
* Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides a strong likelihood that the track is live.
* Loudness: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing the relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
* Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
* Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
* Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, the tempo is the speed or pace of a given piece and derives directly from the average beat duration.

### How to execute
* Run model.py
* Can optimize kernel_width for Gaussian kernel.

### Results

| n        | 7                  | 11                 | 15                 | 25                 | 45                 |
|----------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Default  | 0.627062706270627  | 0.6039603960396039 | 0.6320132013201321 | 0.6072607260726073 | 0.6237623762376238 |
| Inverse  | 0.6171617161716172 | 0.594059405940594  | 0.6320132013201321 | 0.5973597359735974 | 0.6023102310231023 |
| Gaussian | 0.6171617161716172 | 0.594059405940594  | 0.6320132013201321 | 0.5973597359735974 | 0.6023102310231023 |
