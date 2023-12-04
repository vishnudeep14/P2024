import streamlit as st
from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from module import *
import shutil
import imageio
import os

def del_old_files(path):
    for i in os.listdir (path):
        file_path = os.path.join (path, i)
        try:
            if os.path.isfile (file_path):
                os.unlink (file_path)
        except Exception as e:
            st.warning (f"error deleting file {e}")




with st.sidebar:
    # st.image("https://cdn.dribbble.com/users/789882/screenshots/3017138/media/7c45886bb0b8e76ee16647c620211cba.png?resize=800x600&vertical=center")
    # st.image("https://cdn.dribbble.com/users/10549/screenshots/9713618/media/b3f49190adf51bf98108571a65539d91.png?resize=1000x750&vertical=center")
    st.image("https://cdn.dribbble.com/users/8328620/screenshots/15914945/media/055ae9bbaff2a75daa02b2f5cf9f53fb.png?resize=800x600&vertical=center")
    st.title("capstone")
    options=st.radio("Navigate",["upload data","CONVLSTM","GAN","CNN","AQI"])
    st.info("Satellite-Based Tropical Cyclone Intensity Prediction and It's Effect on AQI")

if options=="upload data":
    # st.image("https://wallpapers.com/images/high/cyclone-fo7ob7aiyjvyoeqq.webp")
    # st.image("https://cdn.dribbble.com/users/5521730/screenshots/14270890/media/6acaf4ca3e9e03cb9637ea909d0cf16c.png?resize=1000x750&vertical=center")
    # st.video("https://cdn.dribbble.com/userupload/2609064/file/original-c837b284d8b5db0315b76bca502a3b08.mp4")
    video_url = "https://cdn.dribbble.com/userupload/2609064/file/original-c837b284d8b5db0315b76bca502a3b08.mp4"
    st.write(f'<video src="{video_url}" width="760" height="480" controls autoplay loop>', unsafe_allow_html=True)
    # video_html = f"""
    # <video width="760" height="480" controls autoplay loop>
    #     <source src="{video_url}" type="video/mp4">
    #     Your browser does not support the video tag.
    # </video>
    # """
    # st.markdown(video_html, unsafe_allow_html=True)

    # st.header('<span style="color: red;">Upload Images</span>', unsafe_allow_html=True)
    st.markdown('<h1 style="color: Blue;">Upload Images</h1>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    files_identifier = ",".join ([f"{file.name}_{file.size}" for file in uploaded_files]) if uploaded_files else ""
    if uploaded_files:
        st.success("Images uploaded successfully!")
        st.markdown (f"**Files Identifier:** {files_identifier}")

        path = './uploaded_images/'
        shutil.rmtree(path,ignore_errors=True)
        os.makedirs(path, exist_ok=True)


        for i, file in enumerate (uploaded_files):
            image = Image.open (file)
            image.save (os.path.join (path, f"uploaded_image_{i}.jpg"))


if options == "CONVLSTM":
    # Load and process images
        path="./uploaded_images"
        res = file_load(path)
        ans = make_seq(path, res)
        # img_array = conv_img_array(ans)

        from PIL import Image
        # Path to the directory containing your images
        images_directory = "./prediction"

        # List to store image filenames
        image_files = []

        # Iterate over files in the directory
        for filename in os.listdir(images_directory):
            if filename.endswith(".png"):
                image_files.append(os.path.join(images_directory, filename))

        # Sort the list of image files if needed
        image_files.sort()

        # Create a list to store PIL Image objects
        images = []

        # Open and append each image to the list
        for image_file in image_files:
            img = Image.open(image_file)
            images.append(img)

        # Path to save the resulting GIF
        output_gif_path = "./prediction/output.gif"

        # Save the GIF
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=500,loop=0)

        img_array = conv_img_array(ans)

        if st.button("view sequence image"):
            # st.header("Processed gif Sequence")
            st.image(output_gif_path, caption='gif', use_column_width=True)
            # print_seq(img_array)
        
        if st.button("generate prediction"):
            st.header("Generating Predictions")
            model_path = "E:/CAPSTONE PROJECT/phase2/new_cap/images_seq_final.h5"
            new_data_train = make_train(img_array)
            new_data_test = make_test(img_array)
            output_path = './prediction'
            path_new='./prediction_1'
            img_path = './prediction_1/second_last_predicted_image.png'
            del_old_files(output_path)
            which = 3
            result_track = generate_and_save_predictions(model_path, new_data_train, new_data_test, output_path,which)
            save_prev(new_data_train,model_path,path_new,which)
            st.success("Predictions generated and saved successfully!")
            st.image(img_path)
            st.info("Predicted last frame of ConvLSTM model")

            
        
if options=="GAN":
    st.header("GAN Output")
    path='./prediction_1/second_last_predicted_image.png'
    st.image(path)

if options=="CNN":
    st.header("cnn working")
    path='./prediction_1/second_last_predicted_image.png'
    # st.image(path)
    model_path='./cnn2.h5'
    predict_and_display(path,model_path)


if options=="AQI":
    st.header("AQI Prediction:")
    path='E:/CAPSTONE PROJECT/phase2/new_cap/regression_model.pkl'
    model_load=pickle.load(open(path, 'rb'))
    T= st.number_input ("Temp")
    TM=st.number_input ("Tmax")
    Tm = st.number_input ("Tmin")
    SLP = st.number_input ("slp")
    H=st.number_input ("Humidity")
    VV=st.number_input ("vv")
    V=st.number_input ("v")
    VM=st.number_input ("vm")
    res = 0
    if st.button("Predict"):
        res=aqi_pred(T,TM,Tm,SLP,H,VV,V,VM,model_load)
    st.success("AQI IS{}:".format(res))
    
    st.markdown('<h2 style="color:white">Description of Air Quality</h2>', unsafe_allow_html=True)
    if res>5 and res<=50:
        st.markdown('<h2 style="color: green;">Air quality is satisfactory, and air pollution poses little or no risk.</h2>', unsafe_allow_html=True)
    elif res>=51 and res<=100:
        st.markdown('<h2 style="color: yellow;">Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.</h2>', unsafe_allow_html=True)
    elif res>=101 and res<=150:
        st.markdown('<h2 style="color: orange;">Members of sensitive groups may experience health effects. The general public is less likely to be affected.</h2>', unsafe_allow_html=True)
    elif res>=151 and res<=200:
        st.markdown('<h2 style="color: red;">	Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.</h2>', unsafe_allow_html=True)
    elif res>=201 and res<=300:
        st.markdown('<h2 style="color: purple;">Health alert: The risk of health effects is increased for everyone.</h2>', unsafe_allow_html=True)
    elif res>=301:
        st.markdown('<h2 style="color: maroon;">Health warning of emergency conditions: everyone is more likely to be affected.</h2>', unsafe_allow_html=True)






