import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the training data
print("Loading training data...")
train_data = pd.read_excel("Dataset/Data_Train.xlsx")

print(f"Data loaded: {train_data.shape}")
print("\nData columns:", train_data.columns.tolist())

# Remove missing values
train_data.dropna(inplace=True)

# Feature engineering - Date of Journey
train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data["Journey_month"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.month

# Feature engineering - Departure Time
train_data["Dep_hour"] = pd.to_datetime(train_data.Dep_Time).dt.hour
train_data["Dep_min"] = pd.to_datetime(train_data.Dep_Time).dt.minute

# Feature engineering - Arrival Time
train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

# Feature engineering - Duration
duration = list(train_data["Duration"])
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    if 'h' in duration[i]:
        h = duration[i].split('h')[0]
        m = duration[i].split('h')[1].split('m')[0] if 'm' in duration[i] else '0'
        duration_hours.append(int(h))
        duration_mins.append(int(m))
    else:
        duration_hours.append(0)
        duration_mins.append(int(duration[i].split('m')[0]))

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins

# Handle Total_Stops
train_data['Total_Stops'] = train_data['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

# Encode Airline (One-Hot Encoding)
airline_dummies = pd.get_dummies(train_data['Airline'], prefix='Airline', drop_first=True)
train_data = pd.concat([train_data, airline_dummies], axis=1)

# Encode Source (One-Hot Encoding)
source_dummies = pd.get_dummies(train_data['Source'], prefix='Source', drop_first=True)
train_data = pd.concat([train_data, source_dummies], axis=1)

# Encode Destination (One-Hot Encoding)
destination_dummies = pd.get_dummies(train_data['Destination'], prefix='Destination', drop_first=True)
train_data = pd.concat([train_data, destination_dummies], axis=1)

# Select features for training
feature_cols = ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
                'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins']

# Add airline features
airline_features = [col for col in train_data.columns if col.startswith('Airline_')]
feature_cols.extend(airline_features)

# Add source features
source_features = [col for col in train_data.columns if col.startswith('Source_')]
feature_cols.extend(source_features)

# Add destination features
destination_features = [col for col in train_data.columns if col.startswith('Destination_')]
feature_cols.extend(destination_features)

# Prepare X and y
X = train_data[feature_cols]
y = train_data['Price']

print("\nFeatures being used:", feature_cols)
print(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate model
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")

# Save the model
print("\nSaving model as 'flight_rf.pkl'...")
with open("flight_rf.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Model saved successfully!")
print("\nYou can now run 'python app.py' to start the Flask application.")
