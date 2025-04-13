import tkinter as tk
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from hmm_model import train_hmm, predict_disease

# === Disease-Symptom Map for Follow-up Questions ===
disease_symptoms = {
    "Psoriasis": ["itchy", "scaly", "peeling", "burning"],
    "Flu": ["fever", "chills", "cough", "fatigue"],
    "Cold": ["runny", "sneeze", "congestion", "cough"],
    # Add more diseases and symptoms here
}

# === Load the trained HMM model once ===
file_path = "Symptom2Disease.csv"
model, states, observations = train_hmm(file_path)

# === Track conversation state ===
conversation_state = {
    "awaiting_followup": False,
    "suspected_disease": None,
    "missing_symptoms": [],
    "user_symptoms": []
}

# === Handle user input and bot response ===
def on_send_button_click():
    user_input = user_entry.get().strip()
    if user_input == "":
        return

    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"You: {user_input}\n")

    # Tokenize and filter only valid symptoms
    input_tokens = word_tokenize(user_input.lower())
    valid_tokens = [token for token in input_tokens if token in observations]

    # Add new unique symptoms to the conversation state
    for token in valid_tokens:
        if token not in conversation_state["user_symptoms"]:
            conversation_state["user_symptoms"].append(token)

    # === Follow-up handling ===
    if conversation_state["awaiting_followup"]:
        confirmed = [
            s for s in conversation_state["missing_symptoms"]
            if any(s in word for word in valid_tokens)
        ]
        if confirmed:
            bot_reply = f"Thanks! That confirms more symptoms. Based on your input, you likely have {conversation_state['suspected_disease']}."
        else:
            bot_reply = f"Thanks. Since you're not showing those symptoms, it may not be {conversation_state['suspected_disease']}. Please consider seeing a healthcare professional."

        # Reset state
        conversation_state["awaiting_followup"] = False
        conversation_state["suspected_disease"] = None
        conversation_state["missing_symptoms"] = []
        conversation_state["user_symptoms"] = []

    # === Initial prediction stage ===
    else:
        unique_symptoms = conversation_state["user_symptoms"]
        if len(unique_symptoms) < 3:
            bot_reply = "Could you please describe a few more symptoms so I can better understand your condition?"
        else:
            predicted = predict_disease(model, states, observations, unique_symptoms)
            conversation_state["suspected_disease"] = predicted

            expected_symptoms = disease_symptoms.get(predicted, [])
            missing = [sym for sym in expected_symptoms if sym not in unique_symptoms]

            if missing:
                conversation_state["awaiting_followup"] = True
                conversation_state["missing_symptoms"] = missing
                bot_reply = f"Based on your symptoms, you might have {predicted}. Do you also experience any of the following: {', '.join(missing)}?"
            else:
                bot_reply = f"Based on your symptoms, you might have {predicted}."

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
