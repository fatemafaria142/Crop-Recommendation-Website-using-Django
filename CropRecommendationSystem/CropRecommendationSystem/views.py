import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from django.shortcuts import render


def home(request):
    return render(request, 'homepage.html')


def predict(request):
    return render(request, 'prediction.html')


# Read the CSV file into a DataFrame
df = pd.read_csv("C:/Users/USER/PycharmProjects/MyProject/CropRecommendationSystem/data.csv")

# Step 1: Label Encoding for 'label' column
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Step 2: MinMax Scaling for numerical columns
scaler = MinMaxScaler()
numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 3: Drop 'label' column
X = df.drop('label', axis=1)
y = df['label']

# Step 4: Train-Test Split (80% train, 20% test) with stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create XGBoost model for multiclass classification
model = xgb.XGBClassifier(objective='multi:softmax', num_class=22, random_state=42)

# Train the model
model.fit(X_train, y_train)


def result(request):
       # Get user inputs from the request
        n1 = float(request.GET['n1'])
        n2 = float(request.GET['n2'])
        n3 = float(request.GET['n3'])
        n4 = float(request.GET['n4'])
        n5 = float(request.GET['n5'])
        n6 = float(request.GET['n6'])
        n7 = float(request.GET['n7'])

        # Preprocess the user inputs (scaling, feature engineering, etc.)
        user_inputs = np.array([[n1, n2, n3, n4, n5, n6, n7]])
        scaled_inputs = scaler.transform(user_inputs)

        # Make predictions using the loaded model
        prediction = model.predict(scaled_inputs)

        # Map the predicted label back to the crop name
        label_mapping = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton',
                         7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango',
                         13: 'mothbeans',
                         14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
                         19: 'pomegranate',
                         20: 'rice', 21: 'watermelon'}

        predicted_crop = label_mapping.get(int(prediction[0]), 'Unknown')

        # Pass the prediction result (crop name) to the template

        print(f"Prediction: {predicted_crop}")  # Add this line for debugging
        context = {'predicted_crop': predicted_crop}
        return render(request, 'prediction.html', context)

