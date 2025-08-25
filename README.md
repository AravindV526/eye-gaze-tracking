Full Face & Eye Gaze Tracking with Mask Overlay

This project uses OpenCV, dlib, and NumPy to track the full face and eye gaze direction in real-time using a webcam. It estimates head pose, detects eye closure/gaze direction, and allows applying either a semi-transparent mask or a custom graphic mask (PNG) over the detected face.

✨ Features

✅ Detects faces using dlib’s HOG-based face detector

✅ Tracks 68 facial landmarks with shape_predictor_68_face_landmarks.dat

✅ Estimates head pose (Yaw, Pitch, Roll)

✅ Determines eye gaze direction (Left, Right, Up, Down, Center, Eyes Closed)

✅ Overlays a semi-transparent mask on the face

✅ Supports applying a custom mask image (PNG with transparency)

✅ Works in real-time with webcam feed

📦 Requirements

Install dependencies before running: 
pip install opencv-python dlib numpy 
You also need the shape predictor model:

Download: shape_predictor_68_face_landmarks.dat

Extract and place it in the project directory.

Optional (for custom mask):

Add a mask.png file with transparent background.

▶️ Usage

Run the script: 
python gaze_tracking.py
Controls:

Press q → Quit application

Webcam opens and shows:

Gaze direction (text on screen)

Semi-transparent red face mask

Or your custom mask.png (enable by switching function in code)

🖼 Example Output

Semi-Transparent Mask
A red overlay applied to detected face.

Graphic Mask
Replace face with a custom mask image (e.g., cartoon mask, filter, etc.).

📂 Project Structure 
├── gaze_tracking.py   # Main script
├── shape_predictor_68_face_landmarks.dat   # Landmark model (download separately)
├── mask.png           # Optional custom mask
├── README.md          # Project documentation  
🚀 Future Enhancements

Add support for multiple faces simultaneously

Improve gaze estimation accuracy with deep learning models

Integrate with AR/VR applications

📝 License

This project is for educational and research purposes.
