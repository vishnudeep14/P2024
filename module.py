
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st

import tensorflow
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
from sklearn.model_selection import KFold
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.ticker import PercentFormatter

from keras.models import load_model
import h5py
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import imageio
import os
import glob as glob
import imageio
import os

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

from keras.layers import TimeDistributed
from keras.models import load_model
from keras.models import Sequential
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.layers import Conv3D



import cv2
import os
import glob as glob
def file_load(image_folder):
  image_folder =image_folder
  images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]  # or the format of your images
  images.sort()  # Ensure the images are in the correct order
  return images
#list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(image_folder, x)),os.listdir(image_folder)



def make_seq(image_folder,images):
  sequence_length = 3  # adjust as needed
  image_sequences = []

  for i in range(0, len(images) - sequence_length + 1, sequence_length):
      sequence = []
      for j in range(sequence_length):
          img_path = os.path.join(image_folder, images[i + j])
          img = cv2.imread(img_path)
          sequence.append(img)
          image_sequences.append(sequence)
  return image_sequences

def conv_img_array(ans):
  image_sequences_array = np.array(ans)
  image_sequences_array.shape
  img_gray=np.mean(image_sequences_array,axis=-1)
  gray_seq=np.expand_dims(img_gray,axis=-1)
  gray_seq.shape
  return gray_seq




def print_seq(img_array, num_sequences=3, image_size=2):
    # Display up to the first num_sequences sequences of frames
    for sequence_index in range(min(num_sequences, img_array.shape[0])):
        sequence = img_array[sequence_index]

        # Create a grid of subplots for the frames in the sequence
        num_frames = sequence.shape[0]
        rows = int(num_frames / 3) + 1  # Assuming you want 3 columns, adjust as needed
        cols = 3

        # Adjust the figure size based on the specified image_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * image_size, rows * image_size))

        # Display each frame in the sequence in a subplot
        for frame_index, ax in enumerate(axes.flat):
            if frame_index < num_frames:
                frame = sequence[frame_index, :, :, 0]  # Extract the single channel
                ax.imshow(frame, cmap='gray')
                ax.axis('off')  # Turn off axis labels for cleaner display
                ax.set_title(f'Frame {frame_index + 1}')

        plt.suptitle(f'Sequence {sequence_index + 1}')

        # Display the Matplotlib figure using st.pyplot
        st.pyplot(fig)

        # Add caption below each figure
        st.markdown(f"Caption for Sequence {sequence_index + 1}")

        # Close the Matplotlib figure to release resources
        plt.close(fig)

def make_train(img_arr):
  new_data_train=img_arr
  new_data_train = new_data_train / 255
  return new_data_train

def make_test(img_arr):
    new_data_test=img_arr
    new_data_test = new_data_test / 255
    return new_data_test


def generate_and_save_predictions(model_path, new_data_train, new_data_test, output_path='./prediction',which=1):
    # Load the model
    seq = load_model(model_path)

    # Check whether the specified path exists or not
    is_exist = os.path.exists(output_path)

    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(output_path)
        print("The new directory is created!")

    # Generate predictions
    track = new_data_train[which][:2, ::, ::, ::]
    for j in range(4):
        new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])  # (1, 3, 245, 329, 1)
        new = new_pos[::, -1, ::, ::, ::]  # (1, 245, 329, 1)
        track = np.concatenate((track, new), axis=0)  # adds +1 to the first dimension in each loop cycle
        track2 = new_data_train[which][::, ::, ::, ::]



    plt.axis ("off")
    for i in range(3):
        fig = plt.figure(figsize=(6, 2))

        ax = fig.add_subplot(121)

        if i >= 1:
            ax.text(1, 3, 'Predictions:'+str(i+1), fontsize=5)
        else:
            ax.text(1, 3, 'Initial trajectory:'+str(i+1), fontsize=5)

        toplot = track[i, ::, ::, 0]

        plt.axis("off")
        plt.imshow(toplot, cmap="gray")
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth:'+str(i+1), fontsize=5)

        toplot = track2[i, ::, ::, 0]
        if i >= 1:
            toplot = new_data_test[which][i - 1, ::, ::, 0]

        plt.axis("off")
        plt.imshow(toplot, cmap="gray")
        output_name = os.path.join(output_path, str(i+1) + '_animate.png')
        plt.savefig(output_name)



    return track  # You can modify this based on what information you want to return

# Example usage:

def save_prev(new_data_train,model,path,which=1):
    # Path for saving the images

    seq=load_model(model)
    # Check whether the specified path exists or not
    isExist = os.path.exists (path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs (path)
        print ("The new directory is created!")

    # Index of the record to inspect


    # Extract the trajectory

    track = new_data_train[which][:2, :, :, :]

    # Predict and append new frames to the track
    for j in range (4):
        new_pos = seq.predict (track[np.newaxis, :, :, :, :])
        new = new_pos[::, -1, :, :, :]  # Extract the last predicted frame
        track = np.concatenate ((track, new), axis=0)

    # Display only the last predicted image separately
    last_predicted_image = track[2, :, :, 0]
    second_last_predicted_image = track[1, :, :, 0]
    plt.figure (figsize=(6, 2))

    plt.imshow (last_predicted_image, cmap="gray")
    # plt.imsave('/content/last_1.png',last_predicted_image,cmap='gray')
    plt.axis ("off")
    plt.title ('Last Predicted Image')
    plt.show ()

    # plt.imshow(second_last_predicted_image,cmap="gray")
    plt.imsave (os.path.join (path, 'second_last_predicted_image.png'), second_last_predicted_image, cmap='gray')
    plt.axis ("off")


def predict_and_display(image_path,path):
    # Load your trained CNN model
    model = load_model(path)

    # Load and preprocess the image you want to predict
    image = Image.open(image_path)
    image = image.convert('L')

    # Resize the image using OpenCV
    resized_image = cv2.resize(np.array(image), (250, 250))

    # Reshape the resized image for model input
    image_resized = resized_image.reshape((1, 250, 250, 1))

    # Use the model to make a prediction
    prediction = model.predict(image_resized)
    predicted_wind_speed = prediction[0][0] / 10

    def category_of(predicted_wind_speed):
        if predicted_wind_speed <= 118:
            return 'T. Storm'
        elif 119 <= predicted_wind_speed <= 153:
            return 'Category 1'
        elif 154 <= predicted_wind_speed <= 177:
            return 'Category 2'
        elif 178 <= predicted_wind_speed <= 208:
            return 'Category 3'
        elif 209 <= predicted_wind_speed <= 251:
            return 'Category 4'
        else:
            return 'Category 5'

    category_predicted = category_of(predicted_wind_speed)

    # Use Streamlit to display the image and information
    st.image(image_resized.reshape(250, 250), caption='Resized Image',width=250,use_column_width=True)

    # Display the predicted information
    st.title('Predicted Wind Speed: ' + str(predicted_wind_speed) + ' km/hr')
    st.text('Predicted Category: ' + str(category_predicted))

    if category_predicted == 'T. Storm':
        st.write("The predicted wind speed corresponds to a Tropical Storm.")
        st.write('')
        st.markdown('<h1 style="color: red;">Safety Message: Tropical Storm Warning - Be cautious and stay informed about the weather conditions. Secure outdoor items and prepare for potential flooding.</h1>', unsafe_allow_html=True)
    elif category_predicted == 'Category 1':
        st.write("The predicted wind speed corresponds to a Category 1 cyclone.")
        st.write('')
        st.markdown('<h2 style="color: white;">Minimal - No significant structural damage, can uproot trees and cause some flooding in coastal areas.</h2>', unsafe_allow_html=True)
        st.markdown('<h2 style="color: red;">Safety Message: Category 1 Cyclone Alert - Follow evacuation orders, secure your home, and stay tuned to emergency broadcasts for updates.</h2>', unsafe_allow_html=True)
    elif category_predicted == 'Category 2':
        st.write("The predicted wind speed corresponds to a Category 2 cyclone.")
        st.write('')
        st.markdown('<h2 style="color: white;">Moderate - No major destruction to buildings, can uproot trees and signs. Coastal flooding can occur. Secondary effects can include the shortage of water and electricity.</h2>',unsafe_allow_html=True)
        st.markdown('<h2 style="color: red;">Safety Message: Category 2 Cyclone Alert - Evacuate if instructed, secure your property, and ensure you have emergency supplies.</h2>',unsafe_allow_html=True)
    elif category_predicted == 'Category 3':
        st.write("The predicted wind speed corresponds to a Category 3 cyclone.")
        st.write('')
        st.markdown('<h2 style="color: white;">Extensive - Structural damage to small buildings and serious coastal flooding to those on low lying land. Evacuation may be needed.</h2>',unsafe_allow_html=True)
        st.markdown('<h2 style="color: red;">Safety Message: Category 3 Cyclone Warning - Evacuate immediately if directed. Prepare for extensive damage and power outages.</h2>',unsafe_allow_html=True)
    elif category_predicted == 'Category 4':
        st.write("The predicted wind speed corresponds to a Category 4 cyclone.")
        st.write('')
        st.markdown('<h2 style="color: white;">Extreme - All signs and trees blown down with extensive damage to roofs. Flat land inland may become flooded. Evacuation probable.</h2>',unsafe_allow_html=True)
        st.markdown('<h2 style="color: red;">Safety Message: Category 4 Cyclone Warning - Evacuate and seek sturdy shelter. Expect widespread power outages and severe structural damage.</h2>',unsafe_allow_html=True)
    else:
        st.write("The predicted wind speed corresponds to a Category 5 cyclone.")
        st.write('')
        st.markdown('<h2 style="color: white;">Catastrophic - Buildings destroyed with small buildings being overturned. All trees and signs blown down. Evacuation of up to 10 miles inland</h2>',unsafe_allow_html=True)
        st.markdown('<h2 style="color: red;">Safety Message: Category 5 Cyclone Warning - Evacuate immediately to a safe location. Prepare for catastrophic damage and prolonged power outages.</h2>',unsafe_allow_html=True)



def aqi_pred(T,TM,Tm,SLP,H,VV,V,VM,model):
    pred=model.predict([[T,TM,Tm,SLP,H,VV,V,VM]])
    return  pred





