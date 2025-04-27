from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import threading
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

app = Flask(__name__)

# Load the training dataset for model building
if os.path.exists('datasets/multi_data.csv'):
    multi_data = pd.read_csv('datasets/multi_data.csv')
else:
    raise FileNotFoundError("The dataset 'multi_data.csv' is missing!")

# Extract features and labels for training
X = multi_data.iloc[:, :9]  # First 9 columns as features
y = multi_data.iloc[:, -1]  # Last column as labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

# Load the attack types (class labels) from le2_classes.npy
if os.path.exists('./models/le2_classes.npy'):
    attack_types = np.load('./models/le2_classes.npy', allow_pickle=True)
else:
    raise FileNotFoundError("The 'le2_classes.npy' file is missing!")

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
latest_alert_message = None
monitoring_active = False

# Load real-time data once at start
if os.path.exists('datasets/real_data.csv'):
    real_time_data = pd.read_csv('datasets/real_data.csv')
else:
    raise FileNotFoundError("The 'real_data.csv' file is missing!")

current_index = 0  # Keep track of current row

# Function to monitor the real-time dataset continuously
def monitor_data():
    global latest_prediction, latest_details, latest_alert_message, monitoring_active, current_index
    while True:
        if monitoring_active:
            if current_index >= len(real_time_data):
                current_index = 0  # Restart from beginning if end is reached

            try:
                new_data = real_time_data.iloc[current_index:current_index+1]
                features = new_data.iloc[:, :9].values
                scaled_data = scaler.transform(features)

                prediction = model.predict(scaled_data)
                prediction_proba = model.predict_proba(scaled_data)

                predicted_label = prediction[0]
                predicted_attack_type = label_to_attack_type[predicted_label]

                latest_prediction = f"Predicted type of attack: {predicted_attack_type}"
                latest_details = "\n".join([f"Attack Type: '{label_to_attack_type[i]}' - Probability: {prediction_proba[0][i]:.4f}" for i in range(len(label_to_attack_type))])

                if predicted_attack_type != "Normal":
                    latest_alert_message = f"🚨 Anomaly Detected! Type of Anomaly: <b>{predicted_attack_type}</b>. Please take immediate action! 🚀"
                    monitoring_active = False  # Stop monitoring after detecting an anomaly
                else:
                    latest_alert_message = None

                current_index += 1  # Move to next row only if no anomaly is detected

            except Exception as e:
                latest_prediction = "Error fetching data"
                latest_details = str(e)

            time.sleep(3)  # 3 seconds delay between monitoring
        else:
            time.sleep(1)  # Idle sleep if not monitoring


@app.route('/')
def home():
    return render_template('index.html', prediction_text=latest_prediction, details=latest_details, alert_message=latest_alert_message)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i+1}']) for i in range(9)]
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        predicted_label = prediction[0]
        predicted_attack_type = label_to_attack_type[predicted_label]

        prediction_text = f"Predicted type of attack: {predicted_attack_type}"
        details = "\n".join([f"Attack Type: '{label_to_attack_type[i]}' - Probability: {prediction_proba[0][i]:.4f}" for i in range(len(label_to_attack_type))])

        alert_message = None
        if predicted_attack_type != "Normal":
            alert_message = f"🚨 Anomaly Detected! Type of Anomaly: <b>{predicted_attack_type}</b>. Please take immediate action! 🚀"

        return render_template('index.html', prediction_text=prediction_text, details=details, alert_message=alert_message)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/monitor')
def monitor():
    return render_template('monitor.html', prediction_text=latest_prediction, details=latest_details, alert_message=latest_alert_message)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    return redirect(url_for('monitor'))

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return redirect(url_for('monitor'))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
