import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data_from_directory(self, directory):
        """
        Load MFCC features from directory and extract labels from filenames.
        Format expected: <number>_name_serialno
        """
        features = []
        labels = []
        
        # Walk through directory
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                # Extract the digit from the filename
                match = re.match(r'^(\d+)_', filename)
                if match:
                    digit = int(match.group(1))
                    
                    # Load MFCC feature vector from file
                    filepath = os.path.join(directory, filename)
                    feature = np.loadtxt(filepath)
                    
                    # Flatten the feature if it's multidimensional (more than 1D)
                    if feature.ndim > 1:
                        feature = feature.flatten()
                    
                    features.append(feature)
                    labels.append(digit)
        
        return np.array(features), np.array(labels)
    
    def train(self, features, labels, test_size=0.2, random_state=42):
        """
        Train SVM model with grid search for hyperparameter tuning
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        # Create SVM model with grid search
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1
        )
        
        # Train the model
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Print evaluation results
        print("Best parameters:", grid_search.best_params_)
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Return test accuracy
        return grid_search.score(X_test_scaled, y_test)
    
    def predict(self, mfcc_feature):
        """
        Predict the digit for a new MFCC feature vector
        """
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        
        # Scale the feature
        mfcc_scaled = self.scaler.transform([mfcc_feature])
        
        # Predict digit and probability
        digit = self.model.predict(mfcc_scaled)[0]
        probabilities = self.model.predict_proba(mfcc_scaled)[0]
        
        return digit, probabilities
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler to files"""
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler from files"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


# Usage example
if __name__ == "__main__":
    # Create recognizer
    recognizer = DigitRecognizer()
    
    # Directory containing MFCC files
    mfcc_dir = "./mfcc_mean"
    
    # Load data
    print(f"Loading data from directory: {mfcc_dir}")
    features, labels = recognizer.load_data_from_directory(mfcc_dir)
    print(f"Loaded {len(features)} samples with shape {features[0].shape}")
    
    # Train the model
    print("Training model...")
    accuracy = recognizer.train(features, labels)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model
    recognizer.save_model("digit_svm_model.pkl", "digit_svm_scaler.pkl")
    
    # Example of prediction (replace with actual MFCC feature)
    # Uncomment below to test with a sample from your dataset
    # sample_idx = 0
    # sample_mfcc = features[sample_idx]
    # true_digit = labels[sample_idx]
    # 
    # predicted_digit, probabilities = recognizer.predict(sample_mfcc)
    # print(f"True digit: {true_digit}")
    # print(f"Predicted digit: {predicted_digit}")
    # print(f"Prediction probabilities: {probabilities}")