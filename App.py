import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags

model = tf.keras.models.load_model('model_casia_run3.h5')

labels = ['Fake', 'Real']

def preprocess_image(image):
    temp_filename = 'temp_file_name.jpg'
    
    image.save(temp_filename, 'JPEG', quality=90)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image = ela_image.resize((128, 128))
    
    image_array = np.array(ela_image) / 255.0
    
    preprocessed_image = np.expand_dims(image_array, axis=0)
    
    return preprocessed_image, ela_image

def extract_relevant_metadata(image):
    relevant_metadata = {
        "Software": None,
        "Camera Info": None,
        "Date and Time": None,
        "Dimensions": f"{image.width} x {image.height}",
    }
    
    if hasattr(image, '_getexif'):  
        exif_data = image._getexif()
        if exif_data is not None:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "Software":
                    relevant_metadata["Software"] = value
                elif tag_name == "Model":
                    relevant_metadata["Camera Info"] = value
                elif tag_name == "DateTime":
                    relevant_metadata["Date and Time"] = value
    return relevant_metadata

def main():
    st.set_page_config(page_title='Fake Image Detector')
    st.title("Fake Image Detector")
    st.text("Upload an image to detect fake images and check metadata")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
       
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        preprocessed_image, ela_image = preprocess_image(image)
        with col2:
            st.image(ela_image, caption="ELA Converted Image", use_container_width=True)
        
       
        metadata = extract_relevant_metadata(image)
        st.subheader("Image Metadata:")
        st.write(f"**Modified by Software:** {metadata['Software'] or 'Not available'}")
        st.write(f"**Camera Info:** {metadata['Camera Info'] or 'Not available'}")
        st.write(f"**Date and Time:** {metadata['Date and Time'] or 'Not available'}")
        st.write(f"**Dimensions:** {metadata['Dimensions']}")
        
        prediction = model.predict(preprocessed_image)
        prediction_label = labels[np.argmax(prediction)]
        prediction_confidence = np.max(prediction) * 100

        st.subheader("Prediction:")
        st.write(prediction_label)
        st.subheader("Confidence:")
        st.write(f"{prediction_confidence:.2f}%")

if __name__ == "__main__":
    main()
