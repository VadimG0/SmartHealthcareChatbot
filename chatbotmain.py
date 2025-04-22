import tkinter as tk
from tkinter import ttk
from datetime import datetime

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
try:
    model, tokenizer, label_map = train_biobert(file_path)
    status_message = "Model loaded successfully"
except Exception as e:
    status_message = f"Error loading model: {str(e)}"

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

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Insert user message with timestamp and color
    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"{timestamp} You: {user_input}\n", "user")
    
    # Append to user_text
    conversation_state["user_text"] += " " + user_input

    bot_reply = ""

    # === If user types "done", trigger diagnosis ===
    if user_input.lower() in ["done", "diagnose me", "ready"]:
        try:
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
        except Exception as e:
            bot_reply = f"Error processing diagnosis: {str(e)}"
            status_label.config(text=f"Error: {str(e)}")

    # Follow-up response
    elif conversation_state["awaiting_followup"]:
        confirmed = any(sym in user_input.lower() for sym in conversation_state["missing_symptoms"])
        if confirmed:
            bot_reply = f"That confirms it—likely {conversation_state['suspected_disease']}. Please consult a doctor for a detailed diagnosis."
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

    # Insert bot response with timestamp and color
    chat_window.insert(tk.END, f"{timestamp} Bot: {bot_reply}\n", "bot")
    chat_window.insert(tk.END, "-" * 50 + "\n", "separator")
    chat_window.config(state='disabled')
    chat_window.see(tk.END)  # Auto-scroll to bottom
    
    user_entry.delete(0, tk.END)
    status_label.config(text=status_message)  # Update status

def on_entry_submit(event):
    """Handle Enter key press to submit input."""
    on_send_button_click()

def clear_chat():
    """Clear the chat window."""
    chat_window.config(state='normal')
    chat_window.delete(1.0, tk.END)
    chat_window.config(state='disabled')
    status_label.config(text="Chat cleared")

def on_entry_focus_in(event):
    """Handle entry focus in for placeholder."""
    if user_entry.get() == "Enter symptoms or 'done' to diagnose...":
        user_entry.delete(0, tk.END)
        user_entry.config(fg="#e0e0e0")

def on_entry_focus_out(event):
    """Handle entry focus out for placeholder."""
    if not user_entry.get():
        user_entry.insert(0, "Enter symptoms or 'done' to diagnose...")
        user_entry.config(fg="#888888")

def on_button_hover(event, button, hover_bg, original_bg):
    """Change button background on hover."""
    button.config(bg=hover_bg)

def on_button_leave(event, button, hover_bg, original_bg):
    """Restore button background on leave."""
    button.config(bg=original_bg)

# === UI Setup ===
root = tk.Tk()
root.title("Smart Healthcare Chatbot")
root.geometry("800x600")  # Larger window size
root.configure(bg="#2c2c2c")  # Dark gray background
root.resizable(True, True)

# Custom ttk style for modern look
style = ttk.Style()
style.theme_use("clam")
style.configure("TScrollbar", background="#AD974F", troughcolor="#2c2c2c", arrowcolor="#e0e0e0")
style.map("TScrollbar", background=[("active", "#C4A85A")])

# Title label
title_label = tk.Label(
    root,
    text="Smart Healthcare Chatbot",
    font=("Helvetica", 16, "bold"),
    bg="#2c2c2c",
    fg="#e0e0e0"
)
title_label.pack(pady=10)

# Chat window with scrollbar
chat_frame = tk.Frame(root, bg="#2c2c2c")
chat_frame.pack(padx=10, pady=10, fill="both", expand=True)

chat_window = tk.Text(
    chat_frame,
    height=20,
    width=80,
    font=("Helvetica", 10),
    bg="#3c3c3c",
    fg="#e0e0e0",
    bd=0,
    highlightthickness=1,
    highlightbackground="#555555",
    wrap="word"
)
chat_window.config(state='disabled')
chat_window.pack(side="left", fill="both", expand=True)

# Configure tags for chat window
chat_window.tag_configure("user", foreground="#AD974F", font=("Helvetica", 10, "bold"))
chat_window.tag_configure("bot", foreground="#D5D5D5", font=("Helvetica", 10, "bold"))
chat_window.tag_configure("separator", foreground="#555555")

scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=chat_window.yview, style="TScrollbar")
scrollbar.pack(side="right", fill="y")
chat_window.config(yscrollcommand=scrollbar.set)

# Input frame
input_frame = tk.Frame(root, bg="#2c2c2c")
input_frame.pack(padx=10, pady=10, fill="x")

user_entry = tk.Entry(
    input_frame,
    width=60,
    font=("Helvetica", 10),
    bg="#3c3c3c",
    fg="#888888",
    insertbackground="#e0e0e0",
    bd=0,
    relief="flat"
)
user_entry.insert(0, "Enter symptoms or 'done' to diagnose...")
user_entry.bind("<FocusIn>", on_entry_focus_in)
user_entry.bind("<FocusOut>", on_entry_focus_out)
user_entry.bind("<Return>", on_entry_submit)  # Enter key submits
user_entry.pack(side="left", padx=5)

send_button = tk.Button(
    input_frame,
    text="Send",
    command=on_send_button_click,
    font=("Helvetica", 10, "bold"),
    bg="#AD974F",
    fg="#ffffff",
    bd=0,
    relief="flat",
    activebackground="#C4A85A",
    width=10
)
send_button.pack(side="left", padx=5)
send_button.bind("<Enter>", lambda event: on_button_hover(event, send_button, "#C4A85A", "#AD974F"))
send_button.bind("<Leave>", lambda event: on_button_leave(event, send_button, "#C4A85A", "#AD974F"))

clear_button = tk.Button(
    input_frame,
    text="Clear Chat",
    command=clear_chat,
    font=("Helvetica", 10, "bold"),
    bg="#8A743F",
    fg="#ffffff",
    bd=0,
    relief="flat",
    activebackground="#AD974F",
    width=10
)
clear_button.pack(side="left", padx=5)
clear_button.bind("<Enter>", lambda event: on_button_hover(event, clear_button, "#AD974F", "#8A743F"))
clear_button.bind("<Leave>", lambda event: on_button_leave(event, clear_button, "#AD974F", "#8A743F"))

# Status bar
status_label = tk.Label(
    root,
    text=status_message,
    font=("Helvetica", 9),
    bg="#1c1c1c",
    fg="#888888",
    anchor="w",
    padx=10
)
status_label.pack(fill="x", side="bottom")

root.mainloop()