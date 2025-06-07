import os
import pandas as pd
import tensorflow as tf

def load_and_split(csv_path="dataset.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    
    df = df.sample(frac=1, random_state=random_state)  # shuffle

    X = df.drop(columns=["label"]).values.astype("float32")
    Y = tf.keras.utils.to_categorical(df["label"].values, num_classes=10)

    split = int((1 - test_size) * len(X))
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = Y[:split], Y[split:]
    return test_x.reshape(-1, 28, 28, 1), test_y

# --- Load model ---
model = tf.keras.models.load_model("model_tf/model.keras")  # or "model_tf/model.h5"

# --- Load test data ---
test_x, test_y = load_and_split()[0], load_and_split()[1]

# --- Evaluate ---
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
