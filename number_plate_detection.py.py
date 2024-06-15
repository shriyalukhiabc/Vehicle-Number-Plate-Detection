import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
st.title('Number Plate Recognization')
st.write('\n')
path = st.file_uploader('Upload the Image here')
st.write('\n')
st.sidebar.title('DashBoard')
if st.sidebar.button('Created by'):
    st.sidebar.write('Priyanshi Dobariya')
    st.sidebar.write('Shriya Lukhi')
    st.sidebar.write('Ansh Mangukiya')

# Function to detect number plate and recognize text
def recognize_number_plate(image):

     
        # Convert image to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        
        #Blurring image to reduces the noise 
        blur_car_img= cv2.bilateralFilter(gray,13, 100, 100)
        # Use edge detection to highlight the edges
        edges = cv2.Canny(gray, 30, 200)

    
        # Find contours based on edges detected
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Loop over contours to find the best possible number plate contour
        number_plate_contour = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                number_plate_contour = approx
                break
        
        if number_plate_contour is None:
            return "Number plate not detected", image
        
        # Mask the part other than the number plate
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [number_plate_contour], 0, 255, -1)
        
        # Crop the number plate region
        x, y, w, h = cv2.boundingRect(number_plate_contour)
        number_plate = gray[y:y + h, x:x + w]
        
        # Use OCR to read the number plate
        text = pytesseract.image_to_string(number_plate, config='--psm 8')
        
        return text, number_plate



# File uploader for the image
#uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if st.button('Submit'):
    if path is not None:
        # Read the image
        image = Image.open(path)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Perform number plate recognition
        text, number_plate_image = recognize_number_plate(image)
        
        # Display the result
        st.write(f"Recognized Number Plate: {text}")
        #st.image(number_plate_image, caption='Number Plate', use_column_width=True)
