"""
! This is to actually run the model
^ Check the README for setup instructions
"""

import os
import time
import glob

"""
! MAKE SURE YOU INSTALL ALL OF THE LIBRARIES cause this is not gonna run
"""
import numpy as np
import pandas as pd
import cv2
import pyscreenshot as ImageGrab
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

def realtime_recognizer(model_dir="model_tf", crop_box=(70, 230, 920, 1020)):
    """
    Runs real-time digit recognition using a pre-trained TensorFlow model.

    - Captures a screen region (defined by crop_box)
    - Saves it as an image
    - Preprocesses it
    - Predicts digit using the model
    - Displays result on screen

    Press ESC or Enter to exit.
    """
    model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))

    os.makedirs("img", exist_ok=True)
    print("Starting TensorFlow digit recognizer. Press ESC or Enter to exit.")

    while True:
        try:
            # Grab screen
            img = ImageGrab.grab(bbox=crop_box)
            path = os.path.join("img", "current.png")
            img.save(path)

            # Preprocess image
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
            roi = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
            x = (roi.flatten() > 100).astype("float32").reshape(1, 28, 28, 1)

            # Predict
            probs = model.predict(x) #<- this is probs = model.predict(x)
            digit = np.argmax(probs, axis=1)[0]
            print(f"Prediction: {digit}")

            # Display result
            cv2.putText(frame, f"Prediction: {digit}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("TF Result", frame)

            key = cv2.waitKey(1)
            if key in [27, 13]:  # ESC or Enter
                print("Exiting.")
                break

            time.sleep(1)
        except Exception as e:
            print("Error:", e)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_recognizer()
