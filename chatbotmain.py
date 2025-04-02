import tkinter as tk

# Function to handle user input
def on_send_button_click():
    user_input = user_entry.get()  # Get the user input from the entry widget
    if user_input.strip() != "":
        chat_window.insert(tk.END, f"You: {user_input}\n")  # Display the user input
        bot_response = f"Bot: Here is a potential diagnosis based on your symptoms: {user_input}\n"  # Simulated bot response
        chat_window.insert(tk.END, bot_response)  # Display bot response
        user_entry.delete(0, tk.END)  # Clear the input field

# Create the main window
root = tk.Tk()
root.title("Smart Healthbot") #Create the window title

# Create a scrollable chat window
chat_window = tk.Text(root, height=15, width=50, state='disabled')
chat_window.grid(row=0, column=0, padx=10, pady=10)

# Create an entry widget for user input
user_entry = tk.Entry(root, width=40)
user_entry.grid(row=1, column=0, padx=10, pady=10)

# Create a button to send the input
send_button = tk.Button(root, text="Send", command=on_send_button_click)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
