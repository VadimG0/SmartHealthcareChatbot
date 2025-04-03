import pandas as pd
import numpy as np
from hmmlearn import hmm
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

from data_preprocessing import preprocess_data

def train_hmm(file_path):
    # Load and preprocess data
    symptom_map, diseases = preprocess_data(file_path)
    
    # Define states (diseases) and observations (symptoms)
    states = diseases
    observations = list(symptom_map.keys())
    
    # Initialize HMM with Multinomial emissions for discrete observations
    model = hmm.MultinomialHMM(n_components=len(states), n_iter=100, init_params='ste')
    
    # Prepare training data
    df = pd.read_csv(file_path)
    symptom_sequences = []  # List of observation indices
    lengths = []  # Length of each sequence
    
    for _, row in df.iterrows():
        symptoms = word_tokenize(row['text'].lower())
        # Convert symptoms to observation indices
        obs_indices = [observations.index(s) for s in symptoms if s in observations]
        if obs_indices:  # Only add non-empty sequences
            symptom_sequences.extend(obs_indices)
            lengths.append(len(obs_indices))
    
    if not symptom_sequences:
        raise ValueError("No valid symptom sequences found in the dataset.")
    
    # Reshape data for HMM (n_samples, 1)
    X = np.array(symptom_sequences).reshape(-1, 1)
    
    # Fit the model
    model.fit(X, lengths=lengths)
    
    return model, states, observations

def predict_disease(model, states, observations, user_symptoms):
    # Convert user symptoms to observation indices
    obs_seq = [observations.index(symptom) for symptom in user_symptoms if symptom in observations]
    
    if not obs_seq:
        return "Unknown symptoms provided."
    
    # Use Viterbi algorithm to predict most likely state sequence
    obs_seq = np.array(obs_seq).reshape(-1, 1)
    log_prob, state_seq = model.decode(obs_seq, algorithm="viterbi")
    
    # Return the most likely disease (last state in sequence)
    return states[state_seq[-1]]

if __name__ == "__main__":
    file_path = "Symptom2Disease.csv"
    print("Training HMM model...")
    model, states, observations = train_hmm(file_path)
    print("Model trained successfully.")
    
    # Test prediction
    test_symptoms = ["fever", "cough"]
    print(f"Testing with symptoms: {test_symptoms}")
    disease = predict_disease(model, states, observations, test_symptoms)
    print(f"Predicted Disease: {disease}")