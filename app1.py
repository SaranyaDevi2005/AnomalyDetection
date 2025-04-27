from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import threading
import time
import pygame  # Import pygame for sound playback

app = Flask(__name__)

# Load the training dataset for model building
multi_data = pd.read_csv('datasets/multi_data.csv')

# Extract features and labels for training
X = multi_data.iloc[:, :9]  # First 9 columns as features
y = multi_data.iloc[:, -1]   # Last column as labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

# Save the trained model using pickle
pkl_filename = "./models/xgboost_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load the attack types (class labels) from le2_classes.npy
attack_types = np.load('./models/le2_classes.npy', allow_pickle=True)

# Create a dictionary mapping label numbers to attack types
label_to_attack_type = {
    0: "Analysis",
    1: "Backdoor",
    2: "DoS",
    3: "Exploits",
    4: "Fuzzers",
    5: "Generic",
    6: "Normal",
    7: "Reconnaissance",
    8: "Worms"
}

# Variables to hold the latest prediction
latest_prediction = None
latest_details = None
alarm_playing = False  # Variable to track if the alarm is playing
monitoring_active = False  # Set to False initially; monitoring will start only after the button is clicked

# Function to monitor the real-time dataset continuously
def monitor_data():
    global latest_prediction, latest_details, alarm_playing, monitoring_active
    while True:
        if monitoring_active:
            # Load the real-time dataset
            try:
                real_time_data = pd.read_csv('datasets/real_data.csv')
                new_data = real_time_data.sample(n=1)  # Randomly select one row
                features = new_data.iloc[:, :9].values  # Get the features
                scaled_data = scaler.transform(features)  # Scale the data

                # Make prediction
                prediction = model.predict(scaled_data)
                prediction_proba = model.predict_proba(scaled_data)

                # Map the predicted label to the corresponding attack type
                predicted_label = prediction[0]  # Numeric label
                predicted_attack_type = label_to_attack_type[predicted_label]  # Corresponding attack type

                # Log the predicted attack type for debugging
                print(f"Predicted attack type: {predicted_attack_type}")

                # Format prediction result for display
                latest_prediction = f"Predicted type of attack: {predicted_attack_type}"
                latest_details = "\n".join([f"Attack Type: '{label_to_attack_type[i]}' - Probability: {prediction_proba[0][i]:.4f}"
                                             for i in range(len(label_to_attack_type))])

                # Check if the predicted attack type is an attack (i.e., not "Normal")
                if predicted_attack_type != "Normal" and not alarm_playing:
                    print("Attack detected! Playing alarm.")
                    alarm_playing = True
                    monitoring_active = False  # Stop monitoring
                    threading.Thread(target=play_alarm, daemon=True).start()

            except Exception as e:
                latest_prediction = "Error fetching data"
                latest_details = str(e)

        # Sleep for a specified time before fetching new data (e.g., every 5 seconds)
        time.sleep(5)

def play_alarm():
    """Function to play the alarm sound."""
    global alarm_playing
    alarm_file = r'C:\3rd Year\Machine Learning\AnomalyDetectionApp\alarm.wav'
    
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(alarm_file)
    
    print("Starting alarm playback.")
    pygame.mixer.music.play(-1)  # Play the alarm in a loop

    while alarm_playing:
        time.sleep(1)  # Keep the thread alive while the alarm is playing

    pygame.mixer.music.stop()  # Stop the music when alarm_playing is False

@app.route('/')
def home():
    return render_template('index.html', prediction_text=latest_prediction, details=latest_details)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data for 9 features
        features = [float(request.form[f'feature{i+1}']) for i in range(9)]
        input_data = np.array(features).reshape(1, -1)

        # Scale the input data using the previously fitted scaler
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        # Map the predicted label to the corresponding attack type
        predicted_label = prediction[0]  # Numeric label
        predicted_attack_type = label_to_attack_type[predicted_label]  # Corresponding attack type

        # Format prediction result for display
        prediction_text = f"Predicted type of attack: {predicted_attack_type}"
        details = "\n".join([f"Attack Type: '{label_to_attack_type[i]}' - Probability: {prediction_proba[0][i]:.4f}"
                             for i in range(len(label_to_attack_type))])

        return render_template('index.html', prediction_text=prediction_text, details=details)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/monitor')
def monitor():
    return render_template('monitor.html', prediction_text=latest_prediction, details=latest_details)

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    global alarm_playing, monitoring_active
    alarm_playing = False  # Stop the alarm
    monitoring_active = False  # Stop monitoring
    return render_template('index.html', prediction_text=latest_prediction, details=latest_details)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring_active
    monitoring_active = True  # Start monitoring
    return render_template('monitor.html', prediction_text=latest_prediction, details=latest_details)

if __name__ == '__main__':
    # Start the monitoring thread but it will only run when monitoring_active is True
    threading.Thread(target=monitor_data, daemon=True).start()
    app.run(debug=True)