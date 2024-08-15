# Face-Recognition-with-Attendance-Based-systum
This repository contains a real-time face recognition-based attendance system built using Python, OpenCV, and the K-Nearest Neighbors (KNN) classifier. The system is capable of detecting faces from a video stream, recognizing individuals, and automatically marking their attendance in a CSV file.

Features
Real-time Face Detection: Uses Haar Cascade Classifier to detect faces in a video stream.
Face Recognition: Implements a K-Nearest Neighbors (KNN) model to recognize faces.
Attendance Management: Automatically marks attendance with the name and timestamp, ensuring each person is marked only once per session.
Audio Feedback: Provides audio feedback when attendance is successfully marked.
Prerequisites
Before running the project, ensure you have the following installed:

Python 3.x
OpenCV
NumPy
scikit-learn
pywin32 (for audio feedback on Windows)
Pickle
You can install the required Python packages using pip.

Project Structure
data/: Contains necessary files such as the face detection model (haarcascade_frontalface_default.xml), serialized facial data (faces_data.pkl), and labels (names.pkl).
attendance/: Directory where the attendance CSV files will be saved.
face_recognition_attendance.py: Main script that runs the face recognition and attendance marking process.
How to Run
Prepare the Data:

Collect face images of individuals you want to recognize.
Preprocess and store the facial data and corresponding labels using Pickle to create faces_data.pkl and names.pkl.
Start the System:

Ensure that the data/ folder contains the haarcascade_frontalface_default.xml, faces_data.pkl, and names.pkl files.
Run the face_recognition_attendance.py script using Python.
Usage:

The system will start capturing video from the webcam.
Detected faces will be recognized, and the name of the person will be displayed on the screen.
Attendance will be marked in a CSV file located in the attendance/ directory with the format Attendance_<date>.csv.
Stopping the System:

Press the 'q' key to stop the video capture and close the system.
How It Works
Face Detection: The Haar Cascade classifier is used to detect faces in each frame of the video stream.
Face Recognition: The detected face is resized, flattened, and then passed to a pre-trained KNN model to predict the identity.
Attendance Marking: If the recognized face has not been marked already, the system records the name and timestamp in a CSV file and provides an audio confirmation.
