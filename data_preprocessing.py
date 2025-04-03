import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    symptoms = df['text'].tolist()
    diseases = df['label'].tolist()
    
    tokenized_symptoms = [word_tokenize(symptom.lower()) for symptom in symptoms]
    
    symptom_disease_map = {}
    for tokens, disease in zip(tokenized_symptoms, diseases):
        for token in tokens:
            if token not in symptom_disease_map:
                symptom_disease_map[token] = []
            if disease not in symptom_disease_map[token]:
                symptom_disease_map[token].append(disease)
    
    return symptom_disease_map, list(set(diseases))

if __name__ == "__main__":
    file_path = "Symptom2Disease.csv"  # Update with actual path
    symptom_map, disease_list = preprocess_data(file_path)
    print("Unique Diseases:", disease_list)
    print("Sample Symptom Map:", {k: symptom_map[k] for k in list(symptom_map.keys())[:5]})