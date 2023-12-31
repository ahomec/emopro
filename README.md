# Emotion-based music player
This project utilizes human facial recognition to identify the emotions of the user and then recommend them a song that matches using a music emotion classification algorithm trained with Spotify data gathered from the API.

We utilize a CNN neural network to build the facial emotion recognition model and k-nearest neighbors classification to build the music emotion classification model.

Contributors: [@ahomec](https://github.com/ahomec), [@aliyaliyuan](https://github.com/aliyaliyuan)

### Process diagram
![Process Diagram](imgs/process-diagram.png)

# Human Facial Emotion Recognition
The system is designed to analyze facial expressions in a given video and predict the corresponding emotions. The emotions considered are Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Requirements

Make sure you have the following installed on your system:

numpy==1.21.2
opencv-python==4.5.3.56
keras==2.6.0
tensorflow==2.6.0
matplotlib==3.4.3
seaborn==0.11.2

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/ahomec/emopro
cd Human_Facial_Emotion_Recognition
```

2. Go to "Human_Facial_Emotion_Recognition" (Source: Fernandes, Stephen. “Human_Facial_Emotional_Recognition”, MTH 366 Machine Learning, Spring 2023.) folder in the repository.
4. Download all the files into whatever environment you are running the code in.

## Usage

There are four models: the original, batch normalization, data augmentation, and a learning rate scheduler. Choose whichever model you would like. You can also compare the results as they all have confusion matrices and classification reports. For the README.md purposes, the example uses the oriignal (emo_reco.py). Check the "code" folder in the repository for more. 

Run the emotion recognition program:

```bash
python emo_reco.py
```

The program will process each frame in the specified video and display the detected emotions. Additionally, it will generate a confusion matrix and a classification report based on the true emotions provided.

## Notes

- The program assumes a video format as input, make sure to customize the video file path in the script accordingly.
- Adjust the parameters in the `detectMultiScale` function for face detection according to your video characteristics.
- The true emotions are provided in the script (`true_emotions` list) for evaluation purposes. Adjust it based on your ground truth.

## Training dataset
The model has been pre-trained. See 'Human_Facial_Emotion_Recognition' folder.

If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Classification Report with sample video
![og](https://github.com/ahomec/emopro/assets/107160638/b8b3b572-9e2a-47f2-9c44-5a9aac36827c)

## Sample video 
Ground truth value = "Happy" (3)
https://github.com/ahomec/emopro/assets/107160638/8d1a4313-a708-4171-bffb-4f6a737550ed


## Spotify Music Classification Model

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

![](imgs/k-fold-cv.png)

The optimal parameters for the k-NN model was using inverse weights, with k = 19 and results in an accuracy of 80.10%
