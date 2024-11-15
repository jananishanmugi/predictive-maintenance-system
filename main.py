import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
file_path = 'C://drive1.csv'
data = pd.read_csv(file_path)
print(data.info())
print(data.head())
data = data.drop(columns=['WARM_UPS_SINCE_CODES_CLEARED ()'], errors='ignore')

# Step 1: Handle Missing Values
# Filling missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Step 2: Feature Scaling
# Separating features and target variable (assuming 'ENGINE_RUN_TINE ()' can indicate maintenance needs)
X = data.drop(columns=['ENGINE_RUN_TINE ()'])
y = (data['ENGINE_RUN_TINE ()'] > data['ENGINE_RUN_TINE ()'].median()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=15) # Adjust n_components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Step 5: Model Evaluation
# Predict on the test set and evaluate performance
y_pred = model.predict(X_test_pca)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
import joblib
joblib.dump(model, 'predictive_maintenance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
def predict_maintenance(input_data):
    """
    Function to predict maintenance needs based on new input data.
    """
    # Load the model and scaler
    model = joblib.load('predictive_maintenance_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')

    # Preprocess input data
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)

    # Predict
    prediction = model.predict(input_data_pca)
    return prediction
sample_input = X_test.iloc[:1] # Use one row from the test set
sample_prediction = predict_maintenance(sample_input)
print("Sample Prediction:", "Maintenance Needed" if sample_prediction[0] == 1 else "No Maintenance Needed")
import obd
import pygame
from gtts import gTTS
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from joblib import load
from playsound import playsound

model = load('predictive_maintenance_model.pkl')
scaler = load('scaler.pkl')
pca = load('pca.pkl')
connection = obd.OBD()
def generate_warning_audio():
    jt = gTTS(text="Warning: Maintenance required. Please service your vehicle.", lang='en')
    jt.save("warning.mp3")
    # Plays the audio file; requires a media player that supports mp3
    print("saved")
    playsound("warning.mp3")
def generate_system_ok():
    fk = gTTS(text="Car is fine.You can enjoy your day",lang='en')
    fk.save("system_ok.mp3")
    print("saved")
    playsound("system_ok.mp3")




class MaintenanceApp(App):
    def build(self):
        # Layout and UI setup
        self.layout = BoxLayout(orientation='vertical')
        self.status_label = Label(text="System status: OK", font_size=24)
        self.layout.add_widget(self.status_label)


        # Button to manually trigger check
        self.check_button = Button(text="Run Maintenance Check", font_size=20)
        self.check_button.bind(on_press=self.run_check)
        self.layout.add_widget(self.check_button)

        # Set periodic check every 10 seconds

        return self.layout

    def fetch_vehicle_data(self):
        commands = [obd.commands.RPM, obd.commands.SPEED, obd.commands.COOLANT_TEMP, obd.commands.THROTTLE_POS]
        data = []
        for cmd in commands:
            response = connection.query(cmd)
            data.append(response.value.magnitude if response.value else 0)

        # Ensure we have 25 features, pad with zeros if necessary
        if len(data) < 25:
            data.extend([0] * (25 - len(data)))

        return [data]  # Wrap in a list to form a 2D array

    def run_check(self, instance):


        # Scaling and PCA transformation
        data = self.fetch_vehicle_data()
        print(f"Fetched Data: {data}")


        # Scale and reduce data using PCA
        scaled_data = scaler.transform(data)
        reduced_data = pca.transform(scaled_data)

        # Predict maintenance need
        prediction = model.predict(reduced_data)[0]



        sample_prediction = predict_maintenance(sample_input)

        # Update UI and play warning if maintenance is required
        if sample_prediction[0] == 1:
            self.status_label.text = "Warning: Maintenance Required!"
            generate_warning_audio()
        else:
            self.status_label.text = "System status: OK"
            generate_system_ok()







if __name__ == '__main__':
    MaintenanceApp().run()
for file in ["warning.mp3", "system_ok.mp3"]:
    if os.path.exists(file):
        os.remove(file)