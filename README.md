# Cell Image Disease Classification (EDA)

To run the code, you need to install the following packages:

- tensorflow
- cv2
- os
- shutil
- matplotlib
- zipfile

or

you can use ```environment.yml``` file to create a conda environment.

    conda env create -f environment.yml

for Pip users:

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

## Dataset

Dataset contains 1873 audio files. Each audio file is 3 seconds long and has a sampling rate of 44.1 kHz. The dataset is
divided into 8 classes: angry, calm, disgust, fear, happy, sad, surprise, and neutral. The dataset is available
at [https://zenodo.org/record/1188976#.XqZ2J5NKjIU](https://zenodo.org/record/1188976#.XqZ2J5NKjIU)
this is human voice dataset. for music there is a binary dataset with 2 classes: happy and sad. The dataset is available
at [https://zenodo.org/record/1188976#.XqZ2J5NKjIU](https://zenodo.org/record/1188976#.XqZ2J5NKjIU)
Happy and sad music dataset.

## Feature Extraction

The features are extracted using librosa library. The features are extracted from the audio files and stored in a csv
file. The features are extracted using the following function:

    def extract_feature(file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=audio, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=audio, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T,axis=0)
            return mfccs,chroma,mel,contrast,tonnetz
        except Exception as e:
            print("Error encountered while parsing file: ", file)
            return None, None, None, None, None

### 1.1) What is librosa?

librosa is a python package for music and audio analysis.
It provides the building blocks necessary to create music information
retrieval systems. It is built on top of the scientific
Python stack (numpy, scipy, and matplotlib) and is distributed under the 3-clause BSD license.

### 1.2) What is mfcc?

Mel-frequency cepstral coefficients (MFCCs) are coefficients that
collectively make up an MFC. They are derived from a
type of cepstral representation of the audio clip (a nonlinear
"spectrum-of-a-spectrum").

### 1.3) What is chroma?

Chroma features are a set of features used in music information retrieval.
They are based on the twelve different pitch classes, and are
used to describe the distribution of pitches in a piece of music.

### 1.4) What is mel?

Mel spectrogram is a representation of the short-term power spectrum of a sound, based on a linear
cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

# To run this Project

```bash
source env/bin/activate
cd src
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
cd ..
export PYTHONPATH=./src
uvicorn main:app --reload
```

for cond users:

```bash
conda activate env
cd src
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
cd ..
export PYTHONPATH=./src
uvicorn main:app --reload
```

# To Train the model

```bash
source env/bin/activate
python3 DataLoader.py
python3 train.py --data path/to/audio/file/dir --model model_name
```

for cond users:

```bash
conda activate env
python3 DataLoader.py
python3 train.py --data path/to/audio/file/dir -m model_name
```

## Results and Discussion

Results are saved as confusion matrix and classification report. The model is trained on 80% of the dataset and tested
on 20% of the dataset. The model is trained for 100 epochs. The model is trained using the following function:

    def train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size):
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))
        return model

Discussion: The model is trained on 80% of the dataset and tested on 20% of the dataset. The model is trained for 100
epochs. The model is trained using the following function:

    def train_model(model, train_data, train_labels, test_data, test_labels, epochs, batch_size):
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))
        return model

by adding more data and training the model for more epochs, the accuracy of the model can be increased.

## Deploying into fastapi

```bash
source env/bin/activate
export PYTHONPATH=./src
uvicorn main:app --reload
```

## Yolov5

```bash
source env/bin/activate
cd src
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
cd ..
python src/yolov5/detect.py 
        --weights "src/weights/best.pt" 
      --source data/img --data "src/config/edm8.yaml"
       --name results/test
```

for ffmpeg 
```bash
conda install -c conda-forge ffmpeg
```

# Synopsys

## Title of the Project : Emotion Detection using Audio

## Aim of the Project : To scrape video from Tiktok user and extract audio from the parsed video for classifying emotion of the music. 
