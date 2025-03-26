#Importing and Setingup the necessary tools
from flask import Flask, render_template, Response, jsonify                     #Importing and Setingup the necessary tools 
import cv2
from deepface import DeepFace
import random
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

#Emotion-to-Image Mapping(I have stored the image in the static > images folder according to the picture expression) 
EMOTION_MAP = {
    'happy': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
    'sad': ['image4.jpg', 'image5.jpg', 'image6.jpg'],
    'angry': ['image7.jpg', 'image8.jpg', 'image9.jpg'],
    'fearful': ['image10.jpg', 'image11.jpg', 'image12.jpg'],
    'neutral': ['image13.jpg', 'image14.jpg', 'image15.jpg'],
    'surprised': ['image16.jpg', 'image17.jpg', 'image18.jpg'],
    'confused': ['image19.jpg', 'image20.jpg', 'image21.jpg'],
    'loving': ['image22.jpg', 'image23.jpg', 'image24.jpg'],
    'sleepy': ['image25.jpg', 'image26.jpg', 'image27.jpg'],
    'disgusted': ['image28.jpg', 'image29.jpg', 'image30.jpg']
}
#I have used Global Variables to Stores the most recent detected emotion and to track the frames
last_emotion = 'neutral'
frame_count = 0
analysis_interval = 10 
#I have used the Suggestion Function to Take emotion as input, Checks which images actually exist in the filesystem and to Returns the best image
def get_suggestion(emotion):
    """Get a random local image for the detected emotion"""
    available_images = [
        img for img in EMOTION_MAP.get(emotion, [])
        if os.path.exists(f'static/images/{emotion}/{img}')
    ]
    return {
        'emotion': emotion,
        'image': f'/static/images/{emotion}/{random.choice(available_images)}' if available_images 
                else '/static/images/default.jpg'
    }

#I have used video frame generator to Opens webcam and to Sets resolution
def gen_frames():
    global last_emotion, frame_count
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        if frame_count % analysis_interval == 0:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                last_emotion = results[0]['dominant_emotion']
                cv2.putText(frame, last_emotion, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Analysis error: {e}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_suggestion')
def handle_suggestion():
    return jsonify(get_suggestion(last_emotion))

if __name__ == '__main__':
    app.run(debug=True)                                                                                                                           