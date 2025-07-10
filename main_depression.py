import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Main script for depression vs ME/CFS classification using NNFS neural network."""

import nnfs
nnfs.init()
import numpy as np
import pandas as pd

from layer import Layer_Dense
from activation_function import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy

def main():
    print("depression vs ME/CFS classification")
    
    # Load and preprocess data
    df = pd.read_csv('Datasets/me_cfs_vs_depression_dataset.csv')
    
    # Only keep Depression and ME/CFS
    df = df[df['diagnosis'].isin(['Depression', 'ME/CFS'])]
    
    # Fill missing numeric values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to numeric
    df['pem_present'] = (df['pem_present'] == 1).astype(int)
    df['meditation_or_mindfulness'] = (df['meditation_or_mindfulness'] == 'Yes').astype(int)
    df['gender'] = (df['gender'] == 'Male').astype(int)
    work_mapping = {'Working': 0, 'Partially working': 1, 'Not working': 2}
    social_mapping = {'Very low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very high': 4}
    exercise_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Daily': 4}
    df['work_status'] = df['work_status'].replace(work_mapping).fillna(1)
    df['social_activity_level'] = df['social_activity_level'].replace(social_mapping).fillna(2)
    df['exercise_frequency'] = df['exercise_frequency'].replace(exercise_mapping).fillna(2)
    
    # Prepare features and labels
    X = df.drop(['diagnosis', 'age'], axis=1).to_numpy()
    y = (df['diagnosis'] == 'Depression').astype(int).to_numpy()
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    print(f"data shape: {X.shape}")
    print(f"classes: {len(np.unique(y))}")

    dense1 = Layer_Dense(X.shape[1], 16)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(16, 16)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(16, 2)
    activation3 = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    loss = loss_function.calculate(activation3.output, y)
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    print(f"loss: {loss:.6f}")
    print(f"accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    main() 