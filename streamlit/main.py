import streamlit as st
from PIL import Image
import io
import requests
from utils import read_image, resize_image
import numpy as np
import cv2
from inference import Transformer
transformer = Transformer('streamlit/base.pth')

st.title("Pytorch Anime GAN V2")
st.header("Style Transfer from Landscape to Anime")

url = st.text_input("Enter Image Url:")
path = "streamlit/image/sample_image.jpg"
path_anime = "streamlit/image/anime.jpg"

if url:
    img = resize_image(read_image(url))
    # cv2.imwrite(path, img)
    st.image(img)
    transfering = st.button("Transfer image")
    if transfering:
        st.write("")
        st.write("Transfering...")
        anime_img = (transformer.transform(img) + 1) / 2
        # cv2.imwrite(path_anime, anime_img)
        st.image(anime_img[0])
else:
    st.write("Paste Image URL")