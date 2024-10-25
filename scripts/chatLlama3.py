import json
import os
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, render_template

template = """
Answer the question below:

Here is the conversation history: {context}

Question: {question}

Answer:
"""
#models : 
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def load_conversation_history(filename="conversation_history.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}  # Return an empty dictionary if the file is empty or corrupted
    else:
        # Create the file with an empty JSON object if it doesn't exist
        with open(filename, "w") as file:
            json.dump({}, file)
        return {}

def save_conversation_history(history, filename="conversation_history.json"):
    with open(filename, "w") as file:
        json.dump(history, file, indent=4)

def handle_conversation():
    conversation_history = load_conversation_history()
    context = ""
    if conversation_history:
        for user_input, result in conversation_history.items():
            context += f"\nUser: {user_input}\nAI: {result}"

    print("Welcome to chat with Llama3! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Check if the question has been asked before
        if user_input in conversation_history:
            result = conversation_history[user_input]
            print("Bot:", result)
        else:
            result = chain.invoke({"context": context, "question": user_input})
            print("Bot:", result)
            # Save the new conversation to history
            conversation_history[user_input] = result
        
        context += f"\nUser: {user_input}\nAI: {result}"

    # Save the conversation history when the chat ends
    save_conversation_history(conversation_history)

def ask(question, forget):
    start_time = time.perf_counter()
    if forget != "on":
        conversation_history = load_conversation_history()
    else:
        conversation_history = {}
    context = ""
    if conversation_history:
        for user_input, result in conversation_history.items():
            context += f"\nUser: {user_input}\nAI: {result}"

    #print("Welcome to chat with Llama3! Type 'exit' to quit.")
    #while True:
    user_input = "You: " + question
        
    # Check if the question has been asked before
    if user_input in conversation_history:
        result = conversation_history[user_input]
        #print(result)
    else:
        result = chain.invoke({"context": context, "question": user_input})
        # Save the new conversation to history
        #print(result)
        conversation_history[user_input] = result
        
    context += f"\nUser: {user_input}\nAI: {result}"

    # Save the conversation history when the chat ends
    save_conversation_history(conversation_history)
    end_time = time.perf_counter()
    taken_time = int((end_time - start_time) * 1000)
    return result, taken_time


app = Flask(__name__, template_folder='./')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        forget = request.form.get('forget')
        #print(forget)
        answer, taken_time = ask(question, forget)
        #print(str(taken_time) + " ms")
        return render_template('index.html', answer=answer, calculated_time=taken_time)
    else:
        return render_template('./index.html')





if __name__ == "__main__":
    #handle_conversation()
    app.run(debug=True)

