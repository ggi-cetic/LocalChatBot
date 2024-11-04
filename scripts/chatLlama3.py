import json
import os
import subprocess
import time
import random
import textwrap
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
#model = OllamaLLM(model="llama3")
#prompt = ChatPromptTemplate.from_template(template)
#chain = prompt | model

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

# def handle_conversation():
#     conversation_history = load_conversation_history()
#     context = ""
#     if conversation_history:
#         for user_input, result in conversation_history.items():
#             context += f"\nUser: {user_input}\nAI: {result}"

#     print("Welcome to chat with Llama3! Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
        
#         # Check if the question has been asked before
#         if user_input in conversation_history:
#             result = conversation_history[user_input]
#             print("Bot:", result)
#         else:
#             result = chain.invoke({"context": context, "question": user_input})
#             print("Bot:", result)
#             # Save the new conversation to history
#             conversation_history[user_input] = result
        
#         context += f"\nUser: {user_input}\nAI: {result}"

#     # Save the conversation history when the chat ends
#     save_conversation_history(conversation_history)

def ask(question, forget, model, conv_history_file):
    conv_file = conv_history_file['text']
    model = model['text']
    start_time = time.perf_counter()
    if forget != "on":
        conversation_history = load_conversation_history(filename=conv_file)
    else:
        conversation_history = {}
        tokens = question.split(" ")
        long_tokens = [value for value in tokens if len(value) > 5]
        random_tokens = random.sample(long_tokens, min(4, len(long_tokens)))
        conv_file = "_".join(random_tokens) + ".json"  # e.g., "What_life.txt"
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
        #models : 
        model = OllamaLLM(model=model)
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        result = chain.invoke({"context": context, "question": user_input})
        # Save the new conversation to history
        #print(result)
        conversation_history[user_input] = result
        
    context += f"\nUser: {user_input}\nAI: {result}"

    # Save the conversation history when the chat ends
    save_conversation_history(conversation_history, filename=conv_file)
    end_time = time.perf_counter()
    taken_time = int((end_time - start_time) * 1000)
    return result, taken_time

def get_history(selected=0):
    files = []
    i = 0
    for filename in os.listdir("./"):
    # Check if the file is a JSON file
        if filename.endswith('.json'):
            selected_file = i == selected
            #print(i, selected, selected_file)
            files.append({"value": str(i), "text": filename, "selected": selected_file})
            i +=1
    #print(files)
    return files

def get_models(selected=0):
    process = subprocess.Popen(['ollama', 'ls'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    lines = output.decode('utf-8').split('\n')
    models = []
    i = 0
    for line in lines:
        if line:  # Ignore empty lines
            parts = line.split()
            if ":" in parts[0].strip():
                selected_file = i == selected
                model = parts[0].strip().split(":")[0]
                models.append({"value": str(i), "text": model, "selected": selected_file})
                i +=1
    return models
    
    

app = Flask(__name__, template_folder='./')

@app.route('/', methods=['GET', 'POST'])
def index():
    conv_history = get_history()
    models = get_models()
    if request.method == 'POST':
        question = request.form.get('question')
        forget = request.form.get('forget')
        conversation = request.form.get('mySelect')
        model = request.form.get('modelSelect')
        answer, taken_time = ask(question, forget, models[int(model)], conv_history[int(conversation)])
        conv_history = get_history(selected=int(conversation))
        models = get_models(selected=int(model))
        print(type(answer))
        return render_template('index.html', answer=answer, calculated_time=taken_time, options=conv_history, models=models)
    else:
        return render_template('./index.html', options=conv_history, models=models)


if __name__ == "__main__":
    #handle_conversation()
    app.run(debug=True)

