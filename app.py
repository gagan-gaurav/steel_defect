import streamlit as st
import os
from PIL import Image
import numpy as np

# for windows machine use tensorflow 
# import tensorflow as tf 

# for linux machine use tflite_runtime.interpreter 
import tflite_runtime.interpreter as tflite


# for windows machine 
# interpreter = tf.lite.Interpreter('model2.tflite')    

# for linux machine  
interpreter = tflite.Interpreter('model2.tflite')   
  
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

directory_path = os.getcwd()

image_file1 = "imf1.jpg"
image_file2 = "imf2.jpg"


def process_img(image_path):
    im = Image.open(image_path)
    im = im.resize((120, 120))
    im_arr = np.asarray(im)
    im_arr = im_arr/255.0
    im_arr = im_arr.reshape(1, 120, 120, 3)
    im_arr = im_arr.astype('float32')
    return im_arr


def find_pred(im):
    interpreter.set_tensor(input_details[0]['index'], im)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])
    # res = model.predict(im)
    print(res)
    return np.argmax(res) + 1


hide_st_stylex = """
            <style>
            #upload-an-image {display: none;}
            </style>
            """

st.markdown("<h1 style='text-align: center; color: #bd1816;'>Steel Defect Detection Web-App</h1>",
            unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>A Simple WebApp to demonstrate Transfer Learning Predictions on Steel Dataset</h3>",
            unsafe_allow_html=True)

image_file = st.file_uploader('', type=['jpg', 'png'])

st.markdown("<h5 style='text-align: center; margin:20px;'>Upload an image...</h5>",
            unsafe_allow_html=True)

if(image_file):
    with st.expander('Selected Image', expanded=True):
        st.markdown(hide_st_stylex, unsafe_allow_html=True)
        st.image(image_file, use_column_width='auto')

if image_file and st.button('Predict Defect'):
    image = process_img(image_file)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

st.markdown("<hr style='text-align:center; height:3px;  background-color: #000000;'>",
            unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #bd1816;'>Demo-1 : Image of steel</h4>",
            unsafe_allow_html=True)

st.image(image_file1, use_column_width='auto')

if st.button('Predict Defect Img1', key='1'):
    image = process_img(image_file1)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

st.markdown("<hr style='text-align:center; height:3px;  background-color: #000000;'>",
            unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #bd1816;'>Demo-2 : Image of steel</h4>",
            unsafe_allow_html=True)

st.image(image_file2, use_column_width='auto')

if st.button('Predict Defect Img2', key='2'):
    image = process_img(image_file2)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {display: none;}
            footer {display: none;}
            header {visibility:hidden}
            ::-webkit-scrollbar {display: none;}
            .stApp {
            height: 100vh;
            margin:auto;
            background-repeat: no-repeat;
            background-size: cover;
            position: relative;
            }
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)
