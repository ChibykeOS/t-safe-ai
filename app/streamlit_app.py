import streamlit as st
import cv2
from detect import detect_and_alert  # Import your function

st.title("T-Safe AI Dashboard")

# Webcam input (start small; add video upload later)
run = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)  # 0 = default webcam

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("No camera access")
        break
    
    # Process frame
    annotated_frame, violations = detect_and_alert(frame)
    
    # Show in dashboard
    FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    # Display alerts
    if violations:
        st.warning("\n".join(violations))
    else:
        st.success("No violations")

camera.release()