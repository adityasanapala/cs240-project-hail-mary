import numpy as np
import sys
import os
from svm import DigitRecognizer

def predict_from_file(model_path, scaler_path, mfcc_file_path):
    """
    Predict digit from a new MFCC file
    """
    # Check if file exists
    if not os.path.exists(mfcc_file_path):
        print(f"Error: File {mfcc_file_path} not found")
        return
    
    try:
        # Load MFCC feature from file
        mfcc_feature = np.loadtxt(mfcc_file_path)
        
        # Flatten the feature if it's multidimensional
        if mfcc_feature.ndim > 1:
            print(f"Original MFCC shape: {mfcc_feature.shape}")
            mfcc_feature = mfcc_feature.flatten()
            print(f"Flattened MFCC shape: {mfcc_feature.shape}")
        
        # Load model
        recognizer = DigitRecognizer()
        recognizer.load_model(model_path, scaler_path)
        
        # Predict
        digit, probabilities = recognizer.predict(mfcc_feature)
        
        # Print results
        print(f"\nPredicted digit: {digit}")
        print("\nProbabilities for each digit:")
        for i, prob in enumerate(probabilities):
            print(f"Digit {i}: {prob:.4f}")
            
        # Find top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Digit {idx}: {probabilities[idx]:.4f}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <mfcc_file_path>")
        sys.exit(1)
    
    mfcc_file_path = sys.argv[1]
    model_path = "digit_svm_model.pkl"
    scaler_path = "digit_svm_scaler.pkl"
    
    predict_from_file(model_path, scaler_path, mfcc_file_path)