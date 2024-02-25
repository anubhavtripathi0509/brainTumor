# import tensorflow as tf
# from keras.models import load_model
# from tensorflow.keras.preprocessing import image


# model=load_model('model_braintumor.h5')
# img = cv2.imread('/content/brain_tumor/Training/pituitary_tumor/p (107).jpg')
# img = cv2.resize(img,(150,150))
# img_array = np.array(img)
# img_array.shape
# img_array = img_array.reshape(1,150,150,3)
# img_array.shape
# img = image.load_img('/content/brain_tumor/Training/pituitary_tumor/p (107).jpg')
# plt.imshow(img,interpolation='nearest')
# plt.show()
# a=model.predict(img_array)
# indices = a.argmax()
# print(indices)

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load the pre-trained model
model = load_model('model_braintumor.h5')

# Define a function to predict brain tumor
def predict_tumor(uploaded_file):
    img = np.array(Image.open(uploaded_file).convert('RGB'))
    img = cv2.resize(img,(150,150))
    img_array = np.array(img)
    a = model.predict(img_array.reshape(1, 150, 150, 3))
    indices = a.argmax()
    return indices

def main():
    st.title("Brain Tumor Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform prediction on the uploaded image
        prediction = predict_tumor(uploaded_file)
        st.write("Prediction:", prediction)
        if prediction == 0:
            st.write("The model predicts that the tumor is glioma tumor.")
        elif prediction == 1:
            st.write("The model predicts that the tumor is meningioma tumor.")
        elif prediction == 2:
            st.write("The model predicts that the tumor is no tumor.")
        else:
            st.write("The model predicts that the tumor is pituitary tumor.")

if __name__ == "__main__":
    main()
