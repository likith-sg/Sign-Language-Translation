import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load ResNet50 Model
model_path = "Model/mobilenetv3_sign_language_model.keras"

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Labels
labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]
# Initialize Camera & Hand Detector
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1)

# Image Processing Parameters
offset = 20
imgSize = 224  # ResNet50 requires 224x224 input

print("Press 'q' to exit the camera.")

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)  # Set draw=False for better performance

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure valid cropping dimensions
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Error: Hand cropped image is empty. Try repositioning your hand.")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Resize Keeping Aspect Ratio
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize), interpolation=cv2.INTER_AREA)
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal), interpolation=cv2.INTER_AREA)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Convert Image for Model
        imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgInput = np.expand_dims(imgInput, axis=0)  # Add batch dimension
        imgInput = imgInput.astype(np.float32) / 255.0  # Normalize

        # Predict with ResNet50 Model
        prediction = model.predict(imgInput)

        if prediction.size > 0:
            index = np.argmax(prediction)  # Get predicted label index
            label = labels[index]

            # Draw UI
            cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 200, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Show Cropped and Processed Images
            cv2.imshow('Hand Cropped', imgCrop)
            cv2.imshow('Processed Image', imgWhite)

    cv2.imshow('Sign Language Detector', imgOutput)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
