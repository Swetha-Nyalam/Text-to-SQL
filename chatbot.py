import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import lamini
import threading
from util.db_util import execute_sql
from util.prompts import chain_of_thought

# Set your API key
lamini.api_key = "a6b5f98f73759c6bd5429214bb9eee4a20c4a56f11e411554021206742a7efdd"
model_name = "f4ff4d7c4f25b782f784d91e646d6de7261f69d71d0abcc927f97070b86c1f3d"
llm = lamini.Lamini(model_name=model_name)

messages = []
MAX_MESSAGES = 30

run_sql_button = None


def run_sql(sql_query):
    result_text = execute_sql("uber_rides.db", sql_query)

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"\n{result_text}\n", 'bot')
    chat_area.see(tk.END)
    chat_area.config(state=tk.DISABLED)


def generate_sql(user_question):
    global messages
    global run_sql_button

    if run_sql_button:
        run_sql_button.config(state=tk.DISABLED)

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"You: {user_question}\n", 'user')
    chat_area.see(tk.END)
    chat_area.config(state=tk.DISABLED)

    loading_label.pack()

    try:
        response = llm.generate(chain_of_thought(
            "colgate", "SQLite", user_question), output_type={"sql": "str"})
        sql_response = response.get("sql", "No SQL generated.")
    except Exception as e:
        sql_response = f"Error: {str(e)}"

    messages.append(f"You: {user_question}")
    messages.append(f"Bot: {sql_response}")

    if len(messages) > MAX_MESSAGES * 2:
        messages = messages[-(MAX_MESSAGES * 2):]

    loading_label.pack_forget()

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "\n")
    chat_area.window_create(tk.END, window=bot_image_label)
    chat_area.insert(tk.END, f"  {sql_response}\n", 'bot')

    run_sql_button = tk.Button(
        root, text="Run SQL", command=lambda: run_sql(sql_response))
    chat_area.window_create(tk.END, window=run_sql_button)

    chat_area.see(tk.END)
    chat_area.config(state=tk.DISABLED)


def on_enter(event):
    user_question = question_entry.get()
    if user_question.strip() == "":
        return

    question_entry.delete(0, tk.END)

    threading.Thread(target=generate_sql, args=(
        user_question,), daemon=True).start()


root = tk.Tk()
root.title("Chatbot UI")

root.geometry("900x900")

root.configure(bg='black')

# Load and display the top image
top_image = Image.open("lamini.png")
top_image = top_image.resize(
    (100, 20), Image.Resampling.LANCZOS)
top_photo = ImageTk.PhotoImage(top_image)

top_image_label = tk.Label(root, image=top_photo, bg='black')
top_image_label.pack(pady=10)

chat_area = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, width=120, height=40, state=tk.DISABLED, bg='black', fg='white')
chat_area.tag_configure('user', justify='right')
chat_area.tag_configure('bot', justify='left')
chat_area.pack(pady=10)

question_label = tk.Label(
    root, text="Enter your question:", bg='black', fg='white')
question_label.pack(pady=10)

question_entry = tk.Entry(root, width=70, bg='black', fg='white')
question_entry.pack(pady=10)
question_entry.bind("<Return>", on_enter)

chatbot_image = Image.open("llama.png")
chatbot_image = chatbot_image.resize((30, 30), Image.Resampling.LANCZOS)
chatbot_photo = ImageTk.PhotoImage(chatbot_image)

loading_label = tk.Label(root, image=chatbot_photo, bg='black')
loading_label.pack_forget()

bot_image_label = tk.Label(root, image=chatbot_photo, bg='black')

root.mainloop()
