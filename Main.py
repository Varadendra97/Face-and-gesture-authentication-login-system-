import cv2
import numpy as np
import os
import pandas as pd
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import warnings
import webbrowser
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import smtplib
import random
import string
from email.mime.text import MIMEText
import getpass
import logging
import mediapipe as mp
import re

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename='auth_system.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create directories if they don't exist
os.makedirs('user_data', exist_ok=True)
os.makedirs('user_data/face_encodings', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize Excel file
def init_files():
    if not os.path.exists('user_data/users.xlsx'):
        df = pd.DataFrame(columns=['username', 'email', 'gesture_sequence', 'face_encoding_file', 'is_admin'])
        df.to_excel('user_data/users.xlsx', index=False)

init_files()

# Gesture recognition using MediaPipe Hands
GESTURE_NAMES = {
    0: "fist",
    1: "thumbs_up",
    2: "peace",
    4: "point_up",
    5: "open_hand",
    6: "call_me",
    7: "rock_on",
    8: "victory",
}

GESTURE_INSTRUCTIONS = {
    "fist": "Make a fist with your whole hand",
    "thumbs_up": "Extend your thumb upwards with other fingers closed",
    "peace": "Extend your index and middle fingers (V sign)",
    "point_up": "Extend your index finger upwards with other fingers closed",
    "open_hand": "Extend all fingers with palm facing camera",
    "call_me": "Extend thumb and pinky with other fingers closed (like a phone)",
    "rock_on": "Extend index and pinky fingers with middle/ring fingers closed",
    "victory": "Extend index and middle fingers spread apart (peace sign with space)",
}

def extract_hand_landmarks(frame, hands):
    """Extract hand landmarks using MediaPipe"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    landmarks = results.multi_hand_landmarks[0].landmark
    return [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]

def recognize_gesture(landmarks):
    """Recognize gestures based on hand landmarks"""
    if not landmarks or len(landmarks) < 21:
        return None
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    def distance(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    thumb_dist = distance(thumb_tip, wrist)
    index_dist = distance(index_tip, wrist)
    middle_dist = distance(middle_tip, wrist)
    ring_dist = distance(ring_tip, wrist)
    pinky_dist = distance(pinky_tip, wrist)
    
    def angle(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.arccos(cosine)
    
    if (index_dist < 0.15 and middle_dist < 0.15 and 
        ring_dist < 0.15 and pinky_dist < 0.15 and
        thumb_dist < 0.15):
        return "fist"
    
    if (thumb_dist > 0.2 and 
        index_dist < 0.15 and middle_dist < 0.15 and 
        ring_dist < 0.15 and pinky_dist < 0.15):
        return "thumbs_up"
    
    if (index_dist > 0.2 and middle_dist > 0.2 and 
        ring_dist < 0.15 and pinky_dist < 0.15 and
        angle(index_tip, middle_tip, wrist) < 1.0):
        return "peace"
    
    if (index_dist > 0.25 and 
        middle_dist < 0.15 and ring_dist < 0.15 and pinky_dist < 0.15):
        return "point_up"
    
    if (index_dist > 0.2 and middle_dist > 0.2 and 
        ring_dist > 0.2 and pinky_dist > 0.2 and
        thumb_dist > 0.2):
        return "open_hand"
    
    if (thumb_dist > 0.2 and pinky_dist > 0.2 and 
        index_dist < 0.15 and middle_dist < 0.15 and ring_dist < 0.15):
        return "call_me"
    
    if (index_dist > 0.2 and pinky_dist > 0.2 and 
        middle_dist < 0.15 and ring_dist < 0.15):
        return "rock_on"
    
    if (index_dist > 0.2 and middle_dist > 0.2 and 
        ring_dist < 0.15 and pinky_dist < 0.15 and
        angle(index_tip, middle_tip, wrist) > 1.5):
        return "victory"
    
    return None

def extract_face_features(frame, face_mesh):
    """Extract face features using MediaPipe Face Mesh"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0].landmark
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return features

def send_otp_email(to_email, otp):
    """Send OTP to user's email"""
    sender_email = "bhatraghavendra093@gmail.com"
    sender_password = "hccb laji vxtp irrd"   # Use App Password if 2FA is on
    subject = "Your OTP for Gesture Sequence Access"
    body = f"Your OTP is: {otp}\nThis OTP is valid for 5 minutes."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print("âœ… OTP sent successfully to your email!")
        return True
    except Exception as e:
        print(f"âŒ Error sending OTP email: {e}")
        return False

def generate_otp():
    otp = random.randint(100000, 999999)  # Generate a random 6-digit OTP
    otp = str(otp)
    return otp

def register_user():
    """Register new user with face, email, gesture sequence, and admin status"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        logging.error("Camera failed to open during user registration")
        return False
    
    try:
        users_df = pd.read_excel('user_data/users.xlsx')
    except Exception as e:
        print(f"Error reading user database: {e}")
        logging.error(f"Error reading user database: {e}")
        return False
    
    username = input("Enter new username: ").strip()
    if username in users_df['username'].values:
        print("Username already exists!")
        logging.warning(f"Registration attempt with existing username: {username}")
        return False
    
    email = input("Enter your email address: ").strip()
    if not email or '@' not in email or '.' not in email:
        print("Invalid email address!")
        logging.warning(f"Invalid email address provided: {email}")
        return False
    if email in users_df['email'].values:
        print("Email already registered!")
        logging.warning(f"Email already registered: {email}")
        return False
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        print("Invalid email format!")
        return False
    
    is_admin = input("Register as admin? (y/n): ").strip().lower() == 'y'
    
    print("\nFace Registration - Show your face clearly (front-facing)")
    face_samples = []
    sample_count = 0
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        while sample_count < 10:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                cv2.putText(display_frame, f"Face sample {sample_count+1}/10 - Press 'c'", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                key = cv2.waitKey(1)
                if key == ord('c'):
                    features = extract_face_features(frame, face_mesh)
                    if features:
                        face_samples.append(features)
                        sample_count += 1
                        print(f"Captured face sample {sample_count}/10")
            else:
                cv2.putText(display_frame, "Show face to camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Registration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if sample_count < 10:
        print("Face registration incomplete!")
        logging.error(f"Face registration incomplete for {username}")
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    encoding_file = f"user_data/face_encodings/{username}.pkl"
    try:
        with open(encoding_file, 'wb') as f:
            pickle.dump(face_samples, f)
    except Exception as e:
        print(f"Error saving face encodings: {e}")
        logging.error(f"Error saving face encodings for {username}: {e}")
        return False
    
    print("\nCreate your 5-gesture sequence from these options:")
    print("\n".join([f"{k}: {v} - {GESTURE_INSTRUCTIONS[v]}" for k, v in GESTURE_NAMES.items()]))
    print("\nYou will need to perform each gesture for 2 seconds to register it.")
    
    gesture_sequence = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        
        while len(gesture_sequence) < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            landmarks = extract_hand_landmarks(frame, hands)
            
            if landmarks:
                hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if hand_results.multi_hand_landmarks:
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        display_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
                current_gesture = recognize_gesture(landmarks)
                
                if current_gesture:
                    cv2.putText(display_frame, f"Gesture Recognized: {current_gesture}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        gesture_sequence.append(current_gesture)
                        print(f"Added gesture {len(gesture_sequence)}: {current_gesture}")
                        time.sleep(0.5)
            
            cv2.putText(display_frame, f"Gesture {len(gesture_sequence)+1}/5 - Press 'c'", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Registration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(gesture_sequence) < 5:
        print("Gesture registration incomplete!")
        logging.error(f"Gesture registration incomplete for {username}")
        return False
    
    new_user = pd.DataFrame([{
        'username': username,
        'email': email,
        'gesture_sequence': str(gesture_sequence),
        'face_encoding_file': encoding_file,
        'is_admin': is_admin
    }])
    
    try:
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_excel('user_data/users.xlsx', index=False)
    except Exception as e:
        print(f"Error saving user data: {e}")
        logging.error(f"Error saving user data for {username}: {e}")
        return False
    
    try:
        users_df = pd.read_excel('user_data/users.xlsx')
        if len(users_df) >= 2:
            update_face_recognition_model()
    except Exception as e:
        print(f"Error updating face recognition model: {e}")
        logging.error(f"Error updating face recognition model: {e}")
    
    role = "Admin" if is_admin else "User"
    print(f"\nRegistration successful! Registered as {role}.")
    print("Your gesture sequence:")
    print(" -> ".join(gesture_sequence))
    print("\nRemember this sequence as you'll need to perform it exactly during authentication.")
    logging.info(f"User {username} registered as {role}")
    return True

def update_face_recognition_model():
    """Train or update the face recognition model"""
    try:
        users_df = pd.read_excel('user_data/users.xlsx')
    except Exception as e:
        print(f"Error reading user database: {e}")
        logging.error(f"Error reading user database: {e}")
        return
    
    if len(users_df) < 2:
        print("Need at least 2 users to train the face recognition model.")
        logging.warning("Insufficient users for face recognition model training")
        return
    
    known_face_features = []
    known_face_names = []
    
    for _, row in users_df.iterrows():
        try:
            with open(row['face_encoding_file'], 'rb') as f:
                features_list = pickle.load(f)
                known_face_features.extend(features_list)
                known_face_names.extend([row['username']] * len(features_list))
        except Exception as e:
            print(f"Error loading features for {row['username']}: {e}")
            logging.error(f"Error loading features for {row['username']}: {e}")
    
    if not known_face_features or len(set(known_face_names)) < 2:
        print("Not enough data to train face recognition model.")
        logging.warning("Insufficient data for face recognition model training")
        return
    
    le = LabelEncoder()
    labels = le.fit_transform(known_face_names)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(known_face_features, labels)
    
    model_path = "models/face_recognition_model.pkl"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump((le, clf), f)
        print("Face recognition model updated successfully!")
        logging.info("Face recognition model updated")
    except Exception as e:
        print(f"Error saving face recognition model: {e}")
        logging.error(f"Error saving face recognition model: {e}")

def load_face_recognition_model():
    """Load the trained face recognition model"""
    model_path = "models/face_recognition_model.pkl"
    if not os.path.exists(model_path):
        logging.warning("Face recognition model not found")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            le, clf = pickle.load(f)
        return le, clf
    except Exception as e:
        print(f"Error loading face recognition model: {e}")
        logging.error(f"Error loading face recognition model: {e}")
        return None, None

def authenticate_admin():
    """Authenticate admin with face recognition and gesture sequence"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        logging.error("Camera failed to open during admin authentication")
        return None
    
    try:
        users_df = pd.read_excel('user_data/users.xlsx')
    except Exception as e:
        print(f"Error reading user database: {e}")
        logging.error(f"Error reading user database: {e}")
        return None
    
    if users_df.empty:
        print("No users registered yet!")
        logging.warning("No users registered for admin authentication")
        return None
    
    admin_df = users_df[users_df['is_admin'] == True]
    if admin_df.empty:
        print("No admins registered!")
        logging.warning("No admins registered")
        return None
    
    le, clf = load_face_recognition_model()
    
    if le is None or clf is None:
        print("Face recognition model not trained yet!")
        logging.warning("Face recognition model not trained")
        return None
    
    username = None
    
    print("Admin Face Authentication - Show your face clearly")
    face_detected = False
    start_time = time.time()
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        while time.time() - start_time < 60:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            features = extract_face_features(frame, face_mesh)
            
            if features:
                try:
                    proba = clf.predict_proba([features])[0]
                    best_class_idx = np.argmax(proba)
                    best_prob = proba[best_class_idx]
                    best_name = le.inverse_transform([best_class_idx])[0]
                    
                    if best_prob > 0.7 and users_df[users_df['username'] == best_name]['is_admin'].iloc[0]:
                        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=display_frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        
                        label = f"Admin: {best_name} ({best_prob*100:.1f}%)"
                        cv2.putText(display_frame, label, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, "Press 'n' to continue", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        key = cv2.waitKey(1)
                        if key == ord('n'):
                            username = best_name
                            print(f"Recognized Admin: {username} with {best_prob*100:.1f}% confidence")
                            face_detected = True
                            break
                except Exception as e:
                    print(f"Error during face recognition: {e}")
                    loggi
