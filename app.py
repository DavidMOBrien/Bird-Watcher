import streamlit as st
import os
from utils import *
from PIL import Image

model = None

col1, col2 = st.columns(2)

with col1:
    st.image(Image.open(".\\doc_images\\bird_watcher_logo.png"))

with col2:
    st.title("Bird-Watcher")

left_column, right_column = st.columns(2)

with left_column:
    model_choice = st.selectbox("Select Model:", get_models())

    model = BirdWatcher(model_choice)

with right_column:
    type_choice = st.selectbox("Select Bird Type:", get_bird_types())
    pic_choice = st.selectbox("Select Bird Picture", get_bird_pics(type_choice))

    image_location = f'.\\data\\test\\{type_choice}\\{pic_choice}'

    image = Image.open(f'.\\data\\test\\{type_choice}\\{pic_choice}')

    st.image(image)

    predicted = model.classify(image_location)

    st.write(f'ACTUAL: {type_choice}')
    st.write(f'PREDICTED: {predicted}')

