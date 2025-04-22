import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Default directory for saving/loading the model
MODEL_DIR = "./biobert_saved_model"

class SymptomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def save_model(model, tokenizer, label_map, save_dir=MODEL_DIR):
    """
    Save the trained model, tokenizer, and label map to disk.
    
    Args:
        model: Trained BioBERT model.
        tokenizer: BioBERT tokenizer.
        label_map: Mapping from disease names to indices.
        save_dir: Directory to save the model.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Model, tokenizer, and label map saved to {save_dir}")

def load_model(save_dir=MODEL_DIR):
    """
    Load the saved model, tokenizer, and label map from disk.
    
    Args:
        save_dir: Directory containing the saved model.
    
    Returns:
        tuple: (model, tokenizer, label_map) or None if not found.
    """
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, 'label_map.pkl')):
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        with open(os.path.join(save_dir, 'label_map.pkl'), 'rb') as f:
            label_map = pickle.load(f)
        print(f"Loaded model, tokenizer, and label map from {save_dir}")
        return model, tokenizer, label_map
    return None

def train_biobert(file_path, save_dir=MODEL_DIR):
    """
    Train BioBERT or load a saved model for disease prediction from symptom text.
    
    Args:
        file_path (str): Path to Symptom2Disease.csv dataset.
        save_dir (str): Directory to save/load the model.
    
    Returns:
        tuple: (model, tokenizer, label_map)
    """
    # Try to load saved model
    saved_model = load_model(save_dir)
    if saved_model:
        return saved_model

    # If no saved model, proceed with training
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(file_path)
    df['text'] = df['text'].str.lower()
    
    # Create label mapping
    diseases = sorted(df['label'].unique())
    num_labels = len(diseases)  # Dynamically set num_labels
    label_map = {disease: idx for idx, disease in enumerate(diseases)}
    df['label_idx'] = df['label'].map(label_map)
    
    # Initialize model with correct num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = SymptomDataset(train_df['text'].tolist(), train_df['label_idx'].tolist(), tokenizer)
    test_dataset = SymptomDataset(test_df['text'].tolist(), test_df['label_idx'].tolist(), tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model
    save_model(model, tokenizer, label_map, save_dir)
    
    return model, tokenizer, label_map

def predict_disease(model, tokenizer, label_map, user_text):
    """
    Predict top 5 diseases from user text.
    
    Args:
        model: Trained BioBERT model.
        tokenizer: BioBERT tokenizer.
        label_map: Mapping from indices to disease names.
        user_text: User's symptom description.
    
    Returns:
        list: Top 5 (disease, confidence) predictions.
    """
    inputs = tokenizer(user_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5_indices = torch.argsort(probs, descending=True)[:5]
    predictions = [(list(label_map.keys())[idx.item()], probs[idx].item()) for idx in top5_indices]
    return predictions

if __name__ == "__main__":
    file_path = "Symptom2Disease.csv"
    model, tokenizer, label_map = train_biobert(file_path)
    test_text = "I have a fever and a cough."
    predictions = predict_disease(model, tokenizer, label_map, test_text)
    print("Top 5 Predicted Diseases:")
    for disease, confidence in predictions:
        print(f"{disease}: {confidence:.2f}")