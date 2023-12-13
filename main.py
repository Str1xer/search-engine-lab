import cv2
import numpy as np
from numpy.linalg import norm
from joblib import dump, load
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from PIL import ImageOps, Image

def tokenize(image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    tokens = []

    if (descriptors is not None):
        tokens = loaded_model.predict(descriptors.tolist())

    return tokens

def normalize_tokens(tokens):
    hist = [0 for i in range(512)]
    for i in tokens:
        hist[i] += 1

    return [float(i)/(max(hist) + 0.000001) for i in hist]

sift = cv2.SIFT_create()

loaded_model = load('search_trained.joblib') 

df_images = pd.read_csv("images_base.csv", sep='|')[["pathfile", "vectorized_image"]]

df_images["vectorized_image"] = df_images.vectorized_image.apply(lambda t: json.loads(t))
df_images["vectorized_image"] = [np.array(i).tolist() for i in df_images["vectorized_image"]]
x = df_images.loc[:, "vectorized_image"].values.tolist()

nbrs = NearestNeighbors(n_neighbors=11).fit(x)

def predict(image):
    tokenized_image = tokenize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    normalized_tokens = normalize_tokens(tokenized_image)

    distances, indices = nbrs.kneighbors([normalized_tokens])

    cv2.imshow("Source", image) 

    k = 1
    for j in indices[0][1:]:
        current_image = cv2.imread(df_images.loc[j, "pathfile"])
        st.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        # cv2.imshow(str(k), current_image) 
        k+=1

    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

# src = cv2.imread("./JPEGImages/2007_000876.jpg")
# src = cv2.imread("./JPEGImages/2007_001149.jpg")

# predict(src)

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])

if st.button('Search', type="primary"):
    st.image(uploaded_file, caption='Источник')
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    predict(open_cv_image)

# if uploaded_files is not None:
#     # To read file as bytes:
#     st.image(uploaded_files, caption='Sunrise by the mountains')