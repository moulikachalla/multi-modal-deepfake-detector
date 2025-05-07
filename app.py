# type: ignore
################################################### Libraries #################################
import streamlit as st        # pip install streamlit
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")

# For Images Data
import requests
import os
import numpy as np
import cv2

from PIL import Image # pip install pillow
from io import BytesIO

from tensorflow import keras
from keras.preprocessing.image import img_to_array

# For Temp Files
import tempfile

# For Audio Data
import librosa

# Audio from Video
from moviepy.editor import VideoFileClip
################################### Loading Trained Predictive Models #########################
# load model, set cache to prevent reloading
@st.cache_resource
def load_model():
    cnn = keras.models.load_model('cnn.keras')  # For image 
    rnn = keras.models.load_model("cnn-rnn.keras") # For Video
    ann = keras.models.load_model('ann.keras') # For audio
    return cnn, rnn, ann

with st.spinner("Loading Model...."):
    cnn, rnn, ann = load_model()

########################################## Helper functions ##################################

# ************************************** CNN ***************************************************

def cnnpreprocess(frame):
    # Converting Image Size to Cnn Input Size
    img=cv2.resize(frame,(256,256))
    #scaling to 0 to 1 range
    if(np.max(img)>1):
        img = img/255.0
    img = np.array([img])
    return img

# *************************************************** CNN-RNN *****************************************
# Functions for Pre-Processing Videos

# Taking Centre of the frame where it contains features
def crop_center_square(frame):
    y, x = frame.shape[0:2] # height & width
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Constants -> We can increase or decrease according to analysis
FRAMES = 10
IMG_SIZE = 256

# Video loading ,Taking Frames Per Video & Pre-Processing each frame
def load_video(video, max_frames=FRAMES, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(video)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
        
    # Pad if the video has fewer frames than SEQ_LEN
    while len(frames) < max_frames:
        frames.append(np.zeros((resize[0], resize[1], 3)))  # Black frame padding
        
    return np.array(frames)

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

# ************************************** ANN **********************************************

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs (flatten to 1D)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast = np.mean(spectral_contrast.T, axis=0)
    
    # Combine both features
    return np.hstack((mfccs, spectral_contrast))

################################################ UI ###########################################

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 250px !important;  /* Adjust Sidebar Width */
        min-width: 250px !important;
    }
    /* Justify main content */
    .stMarkdown p {
        text-align: justify !important;
    }
    .Analyze {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Deep Fake", ["Home", "Upload"], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "black"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "black"},
            "nav-link-selected": {"background-color": "blue"},
        }
    )

    if selected == "Upload":
        selected_sub = option_menu(None, ["Image", "Audio", "Video"],
                                   icons=["image", "soundwave", "camera-video"],
                                   menu_icon="list-task", default_index=0,
                                   styles={"container": {"padding": "0!important", "background-color": "black"},
                                           "icon": {"color": "white", "font-size": "15px"}, 
                                           "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "black"},
                                           "nav-link-selected": {"background-color": "green"}
                                           })
        
if selected == 'Home':
    st.subheader(":green[Deep Fake Identification]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write("Deepfakes (a portmanteau of 'deep learning' and 'fake') are images, videos, or audio which are edited or generated using artificial intelligence tools, and which may depict real or non-existent people. They are a type of synthetic media and modern form of a Media prank.")
    cola, colb = st.columns([0.35,0.65])
    with cola:
        st.write("From this project we can classify deep fakes for any given input.")
        colc, cold, cole = st.columns(3)
        with cold:
            st.write(":red[Image]")
            st.write(":red[Audio]")
            st.write(":red[Video]")
        st.write("Video is Combination of both Image & Audio. For the Taken Video We will classify both Frames & Audio")
        st.write(":red[Click Upload for Predictions]")
            
    with colb:
        st.image("https://sosafe-awareness.com/sosafe-files/uploads/2022/08/Comparison_deepfake_blogpost_EN.jpg")

    st.subheader(":green[Project Particulars]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write("Neural Networks are used for classifying given input into :red[Deep Fake or Real]")
    st.write(":violet[CNN For Image:]")
    st.image("CNN.png")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write(":violet[CNN-RNN for Video:]")
    st.image("CNN-RNN.png")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write(":violet[ANN for Audio:]")
    st.image("ANN.jpg")

else:
    if selected_sub == 'Image':
        st.subheader(':red[Upload Image or Enter the Url of Image:]')
        st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
        col1, col2 , col3 = st.columns([0.5,0.05,0.5])
        with col1:
            pic = st.file_uploader("Select Image File", type=['png','jpeg','jpg'])
        with col2:
            st.write(":blue[Or]")
        with col3:
            url = st.text_input("Enter Image Url Here:")

        colx, coly, colz = st.columns(3)
        with coly:
            submit = st.button('Analyze', type="primary")

        if submit:
            col1, col2 = st.columns([0.5,0.5])
            with col1:
                st.text("Given Image:")
                st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                if url!='':
                    response = requests.get(url)
                    st.image(Image.open(BytesIO(response.content)))
                else:
                    image = Image.open(pic)
                    st.image(image)
    
            with col2:
                with st.spinner('Analyzing...'):
                    if url!='':
                        response = requests.get(url)
                        pic  = BytesIO(response.content)
                    else:
                        pic = pic

                    # load and preparing the image
                    image = Image.open(pic)
                    image = img_to_array(image)

                    # Model prediction
                    img = cnnpreprocess(image)
                    prob = round(cnn.predict(img)[0][0],2) # will predict probability
                    cls = ':red[Fake]' if prob<0.5 else ':green[Real]'
                    st.write(f"Probability For Given Image:")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(prob)
                    st.write("Prediction Based on Probablity: (if prob<0.5 -> fake)")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(cls)

    elif selected_sub == 'Video':
        st.subheader(':red[Upload Video (30sec to 1Min: For Quick Analysis):]')
        st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.25,0.5,0.25])
        with col2:
            video = st.file_uploader("Select Video File", type=["mp4", "avi", "mov"])

        colx, coly, colz = st.columns(3)
        with colz:
            submit = st.button('Analyze', type="primary")


        if submit:
            st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
            st.text("Given Video:")
            col1, col2, col3 = st.columns([0.2,0.5,0.2])
            with col2:
                st.video(video)
    
            colx, coly = st.columns(2)
            with colx:
                with st.spinner('Analyzing...'):
                    # Taking Video Path
                    if video is not None:
                        # Saving the uploaded file temporarly
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                            tfile.write(video.read())
                            videopath = tfile.name # Path of saved file
                            # Taking audio path
                            audiopath = videopath.replace('.mp4', '.wav')
                    
                    # Pre-Processing video
                    video_data = load_video(videopath)
                    video_data = np.expand_dims(video_data, axis=0)  # Add batch dimension

                    # Video Prediction
                    prob = round(rnn.predict(video_data)[0][0],2)
                    cls = ':red[Fake]' if prob<0.5 else ':green[Real]'
                    st.write(f"Probability For Given Video:")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(prob)
                    st.write("Prediction Based on Probablity: (if prob<0.5 -> fake)")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(cls)
            with coly:
                with st.spinner('Analyzing...'):
                    # Pre-Processing Audio
                    extract_audio(videopath, audiopath)
                    st.text("Extracted Audio:")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.audio(audiopath)
                    features = extract_features(audiopath).reshape(1, -1)
                    # Audio Prediction
                    prob = round(ann.predict(features)[0][0],2)
                    cls = ':red[Fake]' if prob<0.5 else ':green[Real]'
                    st.write(f"Probability For Given Audio:")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(prob)
                    st.write("Prediction Based on Probablity: (if prob<0.5 -> fake)")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(cls)                    

                    # Cleanup temporary file after processing
                    os.remove(videopath)
                    os.remove(audiopath)

    elif selected_sub == 'Audio':
        st.subheader(':red[Upload Audio (30sec to 1Min: For Quick Analysis):]')
        st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.25,0.5,0.25])
        with col2:
            audio = st.file_uploader("Select Audio File", type=["wav", "mp3", "ogg"])

        colx, coly, colz = st.columns(3)
        with colz:
            submit = st.button('Analyze', type="primary")

        if submit:
            col1, col2 = st.columns([0.5,0.5])
            with col1:
                st.text("Given Audio:")
                st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                st.audio(audio)
    
            with col2:
                with st.spinner('Analyzing...'):
                    if audio is not None:
                        # Saving uploaded audio to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tfile:
                            tfile.write(audio.read())
                            audiopath = tfile.name  # file path

                    # Pre-Processing Audio
                    features = extract_features(audiopath).reshape(1, -1)
                    # Prediction
                    prob = round(ann.predict(features)[0][0],2)
                    cls = ':red[Fake]' if prob<0.5 else ':green[Real]'
                    st.write(f"Probability For Given Audio:")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(prob)
                    st.write("Prediction Based on Probablity: (if prob<0.5 -> fake)")
                    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
                    st.subheader(cls)

                    # Cleanup temporary file after processing
                    os.remove(audiopath)