import tkinter as tk
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')

from biobert_model import train_biobert, predict_disease

# === Disease-Symptom Map for Follow-up Questions ===
disease_symptoms = {
    "Acne": ["pimples", "blackheads", "oily skin", "red spots"],
    "Arthritis": ["joint pain", "stiffness", "swelling", "fatigue"],
    "Bronchial Asthma": ["wheezing", "shortness of breath", "cough", "chest tightness"],
    "Cervical spondylosis": ["neck pain", "stiff neck", "shoulder pain", "headache"],
    "Chicken pox": ["itchy rash", "fever", "blisters", "fatigue"],
    "Common Cold": ["runny nose", "sore throat", "cough", "sneezing"],
    "Dengue": ["high fever", "severe headache", "joint pain", "rash"],
    "Dimorphic Hemorrhoids": ["rectal bleeding", "anal itching", "pain during bowel movements", "swelling"],
    "Fungal infection": ["itchy skin", "red patches", "scaling", "discoloration"],
    "Hypertension": ["headache", "dizziness", "chest pain", "shortness of breath"],
    "Impetigo": ["red sores", "blisters", "itching", "crusting"],
    "Jaundice": ["yellow skin", "fatigue", "dark urine", "abdominal pain"],
    "Malaria": ["fever", "chills", "sweating", "headache"],
    "Migraine": ["severe headache", "nausea", "sensitivity to light", "dizziness"],
    "Pneumonia": ["cough", "fever", "chest pain", "difficulty breathing"],
    "Psoriasis": ["scaly patches", "itching", "red skin", "burning sensation"],
    "Typhoid": ["prolonged fever", "abdominal pain", "weakness", "loss of appetite"],
    "Varicose Veins": ["swollen veins", "leg pain", "heaviness", "itching"],
    "allergy": ["sneezing", "itchy eyes", "runny nose", "rash"],
    "anxiety": ["nervousness", "rapid heartbeat", "sweating", "difficulty concentrating"],
    "bronchitis": ["persistent cough", "mucus production", "chest discomfort", "fatigue"],
    "covid-19": ["fever", "dry cough", "fatigue", "loss of taste or smell"],
    "depression": ["sadness", "fatigue", "loss of interest", "sleep problems"],
    "diabetes": ["increased thirst", "frequent urination", "fatigue", "blurred vision"],
    "drug reaction": ["rash", "itching", "swelling", "difficulty breathing"],
    "ear infection": ["ear pain", "hearing loss", "fever", "discharge"],
    "flu": ["fever", "chills", "cough", "body aches"],
    "gastroesophageal reflux disease": ["heartburn", "chest pain", "regurgitation", "difficulty swallowing"],
    "heart disease": ["chest pain", "shortness of breath", "fatigue", "irregular heartbeat"],
    "irritable bowel syndrome": ["abdominal pain", "bloating", "diarrhea", "constipation"],
    "peptic ulcer disease": ["burning stomach pain", "nausea", "bloating", "heartburn"],
    "sinusitis": ["nasal congestion", "facial pain", "headache", "thick mucus"],
    "strep throat": ["sore throat", "fever", "swollen tonsils", "difficulty swallowing"],
    "tuberculosis": ["persistent cough", "fever", "night sweats", "weight loss"],
    "urinary tract infection": ["burning urination", "frequent urination", "abdominal pain", "cloudy urine"]
}

# === Load BioBERT model once ===
file_path = "Symptom2Disease.csv"
model, tokenizer, label_map = train_biobert(file_path)

# === Track conversation state ===
conversation_state = {
    "awaiting_followup": False,
    "suspected_disease": None,
    "missing_symptoms": [],
    "user_text": ""
}

def on_send_button_click():
    user_input = user_entry.get().strip()
    if not user_input:
        return

    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"You: {user_input}\n")
    # Append to user_text
    conversation_state["user_text"] += " " + user_input

    bot_reply = ""

    # === If user types "done", trigger diagnosis ===
    if user_input.lower() in ["done", "diagnose me", "ready"]:
        predictions = predict_disease(model, tokenizer, label_map, conversation_state["user_text"])
        top_disease = predictions[0][0]
        top_confidence = predictions[0][1]
        conversation_state["suspected_disease"] = top_disease

        # Confidence threshold for reliable prediction
        confidence_threshold = 0.4

        if top_confidence < confidence_threshold:
            # Low confidence: show top 3 diseases and ask for more symptoms
            top_3 = predictions[:3]
            bot_reply = f"The diagnosis is not certain (confidence: {top_confidence:.2f}). Possible diseases:\n"
            for disease, confidence in top_3:
                bot_reply += f"- {disease}: {confidence:.2f}\n"
            bot_reply += f"Please provide more symptoms if possible."
            conversation_state["awaiting_followup"] = False  # Stay in symptom collection mode
        else:
            # High confidence: proceed with follow-up questions
            expected = disease_symptoms.get(top_disease, [])
            if expected:
                bot_reply = f"I think you might have {top_disease} (confidence: {top_confidence:.2f}). Do you also have: {', '.join(expected)}?"
            else:
                bot_reply = f"I think you might have {top_disease} (confidence: {top_confidence:.2f}). Please consult a doctor for a detailed diagnosis."
            conversation_state["awaiting_followup"] = True
            conversation_state["missing_symptoms"] = expected

    # Follow-up response
    elif conversation_state["awaiting_followup"]:
        confirmed = any(sym in user_input.lower() for sym in conversation_state["missing_symptoms"])
        if confirmed:
            bot_reply = f"That confirms it—likely {conversation_state['suspected_disease']}."
        else:
            bot_reply = f"If those symptoms aren’t present, it might not be {conversation_state['suspected_disease']}. See a doctor."
        
        # Reset state
        conversation_state["awaiting_followup"] = False
        conversation_state["suspected_disease"] = None
        conversation_state["missing_symptoms"] = []
        conversation_state["user_text"] = ""

    # Symptom collection
    else:
        bot_reply = "Noted. Please provide detailed symptoms (e.g., 'persistent cough for a week') and say 'done' when ready for a diagnosis."

    chat_window.insert(tk.END, f"Bot: {bot_reply}\n")
    chat_window.config(state='disabled')
    user_entry.delete(0, tk.END)

# === UI Setup ===
root = tk.Tk()
root.title("Smart Healthbot")

chat_window = tk.Text(root, height=15, width=50)
chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
chat_window.config(state='disabled')

user_entry = tk.Entry(root, width=40)
user_entry.grid(row=1, column=0, padx=10, pady=10)

send_button = tk.Button(root, text="Send", command=on_send_button_click)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
