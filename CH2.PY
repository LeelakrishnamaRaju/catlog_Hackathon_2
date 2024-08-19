import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Sample dataset (you should have a more extensive dataset for better accuracy)
data = pd.DataFrame({
    'Soil_Type': ['Loamy', 'Clayey', 'Sandy', 'Loamy', 'Sandy'],
    'pH': [6.5, 5.5, 7.0, 6.8, 5.2],
    'Nitrogen': [18, 20, 10, 22, 15],
    'Phosphorus': [12, 14, 8, 15, 9],
    'Potassium': [22, 25, 15, 24, 18],
    'Crop': ['Wheat', 'Rice', 'Maize', 'Soybean', 'Potato']
})

# Convert categorical variable 'Soil_Type' to numerical form
encoder = OneHotEncoder()
encoded_soil = encoder.fit_transform(data[['Soil_Type']]).toarray()
encoded_soil_df = pd.DataFrame(encoded_soil, columns=encoder.get_feature_names_out(['Soil_Type']))
data = pd.concat([data.drop('Soil_Type', axis=1), encoded_soil_df], axis=1)

# Features and labels
X = data.drop('Crop', axis=1)
y = data['Crop']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to recommend crop based on user input
def recommend_crop_ml(input_soil, model, encoder):
    # Preprocessing the input data
    soil_type_encoded = encoder.transform([[input_soil['Soil_Type']]]).toarray()
    input_features = [
        input_soil['pH'], input_soil['Nitrogen'], input_soil['Phosphorus'], input_soil['Potassium']
    ] + list(soil_type_encoded.flatten())
    input_df = pd.DataFrame([input_features], columns=X.columns)
    
    # Predict the crop
    predicted_crop = model.predict(input_df)
    return predicted_crop[0]

# Taking input from the user
input_soil = {
    'Soil_Type': input("Enter the Soil Type (Loamy, Clayey, Sandy): "),
    'pH': float(input("Enter the pH level of the soil: ")),
    'Nitrogen': float(input("Enter the Nitrogen content (in mg/kg): ")),
    'Phosphorus': float(input("Enter the Phosphorus content (in mg/kg): ")),
    'Potassium': float(input("Enter the Potassium content (in mg/kg): "))
}

# Recommend crop based on user input using the trained model
recommended_crop = recommend_crop_ml(input_soil, model, encoder)

# Displaying the recommended crop
print(f"\nRecommended Crop for your soil: {recommended_crop}")
