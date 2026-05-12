import json
import asyncio
import os
import subprocess
import time
import random
from pathlib import Path
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from urllib.parse import urlparse

from flask import Flask, request, render_template

# URL de base Ollama (pour Docker: host.docker.internal:11434)
OLLAMA_BASE_URL = f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}"


def _load_app_config() -> dict:
    """Load configuration from app_config.json with fallback defaults."""
    config_path = Path(os.environ.get("APP_CONFIG_PATH", Path(__file__).parent / "app_config.json"))
    default_config = {
        "app": {
            "default_template": "index3.html",
            "port": 5003,
            "host": "0.0.0.0",
            "debug": True,
        },
        "ui": {
            "default_model_index": 0,
            "default_backend": "searxng",
            "default_conversation_index": 0,
        },
        "agent": {
            "max_iterations": 10,
            "conversation_history_size": 5,
        },
    }
    
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                print(f"✅ Configuration loaded from {config_path}")
                return config
    except Exception as e:
        print(f"⚠️ Error loading {config_path}: {e}")
    
    print("ℹ️ Using default configuration")
    return default_config


# Load configuration
APP_CONFIG = _load_app_config()

# Configuration values with environment fallback
DEFAULT_TEMPLATE = os.environ.get("DEFAULT_TEMPLATE", APP_CONFIG["app"]["default_template"])
APP_PORT = int(os.environ.get("APP_PORT", APP_CONFIG["app"]["port"]))
APP_HOST = os.environ.get("APP_HOST", APP_CONFIG["app"]["host"])
APP_DEBUG = os.environ.get("APP_DEBUG", str(APP_CONFIG["app"]["debug"])).lower() in ("true", "1", "yes")

DEFAULT_BACKEND = os.environ.get("DEFAULT_BACKEND", APP_CONFIG["ui"]["default_backend"])
DEFAULT_MODEL_INDEX = int(os.environ.get("DEFAULT_MODEL_INDEX", APP_CONFIG["ui"]["default_model_index"]))
DEFAULT_CONVERSATION_INDEX = int(os.environ.get("DEFAULT_CONVERSATION_INDEX", APP_CONFIG["ui"]["default_conversation_index"]))

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", APP_CONFIG["agent"]["max_iterations"]))

SEARCH_CONFIG = APP_CONFIG.get("search", {})
SEARXNG_CONFIG = SEARCH_CONFIG.get("searxng", {})
SEARXNG_HEALTH_URL_DEFAULT = SEARXNG_CONFIG.get("health_url", "http://127.0.0.1:5002")
SEARXNG_HEALTH_TIMEOUT_DEFAULT = float(SEARXNG_CONFIG.get("health_check_timeout", 2.0))


template = """
Answer the question from the user below in the most helpful way possible.
Don't flatter him on the information it provides, just answer the question and provide sources url at the end if you have them.
Tools may be available. Don't hesitate to use them to provide an accurate answer.
Question: {question}
Here is the conversation history with the last exchange first: {context}

"""

# Base path of the application
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Init agent mode with langchain_mcp_adapters
async def init_mcp_client():
    """Initialize MCP client and retrieve tools."""
    mcp_script_path = os.path.join(APP_DIR, "search", "search_mcp.py")
    client = MultiServerMCPClient({
        "search_mcp": {
            "transport": "stdio",
            "command": "python3",
            "args": [mcp_script_path],
        }
    })
    
    # MultiServerMCPClient connects automatically; no need to call connect().
    # Retrieve tools directly.
    tools = await client.get_tools()
    
    return client, tools

def load_conversation_history(filename="conversation_history.json"):
    filename = "history/" + filename
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                hist = json.load(file)
                reversed_keys = list(hist.keys())[::-1]  # Get keys and reverse them
                reversed_hist = {}
                for key in reversed_keys:
                    reversed_hist[key] = hist[key]  # Store as (prompt, response)
                return reversed_hist
            except json.JSONDecodeError:
                return {}  # Return an empty dictionary if the file is empty or corrupted
    else:
        # Create the file with an empty JSON object if it doesn't exist
        with open(filename, "w") as file:
            json.dump({}, file)
        return {}

def save_conversation_history(history, filename="conversation_history.json"):
    filename = "history/" + filename
    with open(filename, "w") as file:
        json.dump(history, file, indent=4)

async def ask_with_tools(model, context, user_input, tools, search_backend="auto"):
    """Use the LLM with MCP tools - enhanced agent loop."""
    try:
        # Create LLM with tools
        llm = ChatOllama(model=model, temperature=0, base_url=OLLAMA_BASE_URL)
        llm_with_tools = llm.bind_tools(tools)
        
        # Prepare initial messages
        messages = [
            SystemMessage(content=f"""You are a helpful research assistant with access to web search and content fetching tools.
                                    IMPORTANT INSTRUCTIONS:
                                    1. When asked about recent news, current information or subject you're not sure to be up-to-date, 
                                        **ALWAYS** use the tool search_web first, pick the most relevant URL and use the tool fetch_url_content to get detailed information
                                    2. Use the fetched content to provide a comprehensive answer
                                    3. Include source URLs in your final answer
                                    4. The workflow can be iterative. Only stop when you have enough information to answer the question completely
                                    5. Always pass the backend argument to search_web and use this exact value: {search_backend}
                                    **don't create urls, only use the ones from the search_web tool**
                                    
                                    EXAMPLE WORKFLOW:
                                    User: "What are the latest AI news?"
                                    Step 1 - Call search_web tool:
                                        {{"name": "search_web", "arguments": {{"query": "latest AI news 2026", "backend": "{search_backend}"}}}}
                                    Step 2 - From search results, pick relevant URL and call fetch_url_content:
                                        {{"name": "fetch_url_content", "arguments": {{"url": "https://example.com/ai-news-article"}}}}
                                    Step 3 - Synthesize the fetched content into a comprehensive answer with source URLs.
                                    
                                    Conversation history: {context}"""),
            HumanMessage(content=user_input)
        ]
        
        max_iterations = MAX_ITERATIONS
        iteration = 0
        
        print(f"\n{'='*60}")
        print(f"🤖 Starting agent loop (max {max_iterations} iterations)")
        print(f"{'='*60}\n")
        
        # Agent loop
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            
            # Invoke the LLM
            response = await llm_with_tools.ainvoke(messages)
            
            # Check if the LLM wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 LLM wants to use {len(response.tool_calls)} tool(s)")
                
                # Append LLM response to messages
                messages.append(response)
                
                # Execute all requested tools
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call.get('id', 'unknown')

                    if tool_name == 'search_web':
                        if not isinstance(tool_args, dict):
                            tool_args = {'query': str(tool_args)}
                        tool_args['backend'] = search_backend
                    
                    print(f"   🛠️  Tool: {tool_name}")
                    print(f"   📥 Args: {tool_args}")
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                # If calling fetch_url_content, validate URL
                                if tool_name == 'fetch_url_content' and isinstance(tool_args, dict):
                                    href = tool_args.get('url') or tool_args.get('href') or tool_args.get('link')
                                    if href:
                                        parsed = urlparse(href)
                                        if parsed.scheme not in ('http','https') or not parsed.netloc:
                                            # Instead of raising and losing traceability, append a ToolMessage explaining rejection
                                            err_msg = json.dumps({"error": "invalid_url", "url": href, "reason": "invalid_scheme_or_netloc"}, ensure_ascii=False)
                                            print(f"   ⚠️ Rejected invalid URL for fetch: {href}")
                                            messages.append(ToolMessage(content=err_msg, tool_call_id=tool_call.get('id','invalid-url')))
                                            tool_result = err_msg
                                            raise ValueError(f"Invalid URL passed to fetch_url_content: {href}")
                                tool_result = await tool.ainvoke(tool_args)
                                print(f"   ✅ Result: {len(str(tool_result))} characters")
                            except Exception as e:
                                tool_result = f"Error executing tool: {str(e)}"
                                print(f"   ❌ Error: {e}")
                            break
                    
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"
                    
                    # Append tool result to messages
                    # ToolMessage format for LangChain
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    ))
                
                print(f"   💬 Asking LLM to continue with tool results...")
                
            else:
                # LLM returned a final answer without requesting a tool
                # But sometimes it prints a JSON tool call in plain text
                # Detect and execute JSON-like tool calls emitted as plain text
                content_text = getattr(response, 'content', '') or ''
                parsed_tool = None
                try:
                    # First try to parse it directly
                    try:
                        parsed = json.loads(content_text)
                    except Exception as e:
                        print(f"Direct parse NOK : {e}")
                        parsed = None

                    # Then prepare the response before parsing
                    if parsed is None:
                        # Strip common code-fence markers and whitespace
                        txt = content_text.strip()
                        if txt.startswith('```') and txt.endswith('```'):
                            txt = '\n'.join(txt.split('\n')[1:-1]).strip()
                        if txt.find('"') >= 0:
                            print(f"txt.find : {txt.find('"')}")
                            txt.replace('"', "'")
                        # Try JSON parse
                        try:
                            parsed = json.loads(txt)
                        except Exception as e:
                            print(f"Prepare and parse NOK : {e}")
                            parsed = None

                    # If not JSON, try to find an object-like substring (JSON or Python literal)
                    if parsed is None:
                        import re, ast
                        # Match the first {...} block
                        m = re.search(r"\{[\s\S]*?\}+", content_text)
                        if m:
                            candidate = m.group(0)
                            # Try JSON or python literal
                            try:
                                parsed = json.loads(candidate)
                            except Exception as e:
                                print(f"Regex parse NOK : {e}")
                                try:
                                    parsed = ast.literal_eval(candidate)
                                except Exception as e:
                                    print(f"ast literal parse NOK : {e}")
                                    parsed = None

                    # If still not parsed, final fallback attempt
                    if parsed is None:
                        extracted = content_text[content_text.find("{"):content_text.rfind("}")+1]
                        if extracted:
                            try:
                                parsed = json.loads(extracted)
                            except Exception:
                                parsed = None

                    if isinstance(parsed, dict) and ('name' in parsed or 'tool' in parsed):
                        parsed_tool = parsed
                except Exception:
                    parsed_tool = None

                if parsed_tool:
                    # Normalize parsed structure
                    tool_name = parsed_tool.get('name') or parsed_tool.get('tool')
                    tool_args = parsed_tool.get('parameters') or parsed_tool.get('args') or parsed_tool.get('arguments') or {}

                    # If tool_args contains search_results as a serialized list/JSON, try to parse it
                    if isinstance(tool_args, dict) and 'search_results' in tool_args:
                        sr = tool_args['search_results']
                        if isinstance(sr, str):
                            # Try JSON
                            try:
                                parsed_sr = json.loads(sr)
                                tool_args['search_results'] = parsed_sr
                            except Exception:
                                # Try Python literal
                                try:
                                    import ast
                                    parsed_sr = ast.literal_eval(sr)
                                    tool_args['search_results'] = parsed_sr
                                except Exception:
                                    # leave as string
                                    pass
                    print(f"� Detected textual tool call -> executing: {tool_name} with {tool_args}")
                    # Append model response, then the tool result
                    messages.append(response)
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                # If calling fetch_url_content, validate URL
                                if tool_name == 'fetch_url_content' and isinstance(tool_args, dict):
                                    href = tool_args.get('url') or tool_args.get('href') or tool_args.get('link')
                                    if href:
                                        parsed = urlparse(href)
                                        if parsed.scheme not in ('http','https') or not parsed.netloc:
                                            err_msg = json.dumps({"error": "invalid_url", "url": href, "reason": "invalid_scheme_or_netloc"}, ensure_ascii=False)
                                            print(f"   ⚠️ Rejected invalid URL for textual fetch: {href}")
                                            messages.append(ToolMessage(content=err_msg, tool_call_id=parsed_tool.get('id','invalid-url')))
                                            tool_result = err_msg
                                            raise ValueError(f"Invalid URL passed to fetch_url_content: {href}")
                                tool_result = await tool.ainvoke(tool_args)
                                print(f"   ✅ Executed textual tool: {tool_name} -> {len(str(tool_result))} chars")
                            except Exception as e:
                                tool_result = f"Error executing tool: {str(e)}"
                                print(f"   ❌ Error executing textual tool: {e}")
                            break
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=parsed_tool.get('id','text-fallback')))
                    # Continue loop to let LLM process the tool result
                    print("   💬 Continuing agent loop after textual tool execution")
                    continue

                # Otherwise treat as final answer
                print(f"✅ LLM provided final answer (no more tools needed)")
                print(f"📝 Answer length: {len(content_text)} characters")
                return content_text
        
        # If max iterations is reached
        print(f"\n⚠️  Reached maximum iterations ({max_iterations})")
        print(f"📝 Returning last response")
        return response.content if hasattr(response, 'content') else str(response)
            
    except Exception as e:
        print(f"❌ Error with tools: {e}")
        import traceback
        traceback.print_exc()
        return None

def ask(question, forget, model, conv_history_file, tools, search_backend="auto"):
    resultlist = ["\n\n------------------- Conversation history -------------------"]
    # Define inputs : conv_file & model
    conv_file = conv_history_file['text']
    model = model['text']
    # Save start_time for computation time
    start_time = time.perf_counter()
    # if it's not a new conversation, load the conversation
    if forget != "on":
        conversation_history = load_conversation_history(filename=conv_file)
        for key, value in conversation_history.items():
            resultlist.append(key)
            resultlist.append("\nChatbot: " + value)
            resultlist.append("------------------------------------------------------------") 
    # Else create the file to save the history
    else:
        conversation_history = {}
        tokens = question.split(" ")
        long_tokens = [value for value in tokens if len(value) > 6]
        random_tokens = random.sample(long_tokens, min(4, len(long_tokens)))
        conv_file = "_".join(random_tokens)  # e.g., "What_life.txt"
        conv_file = ''.join(c for c in conv_file if c.isalnum() or c == '_') + ".json"
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
        # If MCP tools are available, use agent mode; otherwise use simple chain mode
        if tools:
            print(f"Using MCP tools mode with {model}") 
            # Use MCP tools asynchronously
            result = asyncio.run(ask_with_tools(model, context, user_input, tools, search_backend=search_backend))
            
            # If tool mode fails, fallback to simple mode
            if result is None:
                print("⚠️ Fallback to chatbot mode (no MCP tools)")
                llm_model = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL)
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm_model
                result = chain.invoke({"context": context, "question": user_input})
        else:
            print("Using simple chain mode (no MCP tools)")
            # Use OllamaLLM for simple mode
            llm_model = OllamaLLM(model=model, base_url=OLLAMA_BASE_URL)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm_model
            result = chain.invoke({"context": context, "question": user_input})
        
        # Save new conversation to history
        print(result)
        conversation_history[user_input] = result
        
    context += f"\nUser: {user_input}\nAI: {result}"
    # Add result to resultlist
    resultlist.insert(0, "\nChatbot: " + result)
    resultlist.insert(0, user_input)
    resultlist.insert(0,"----------------------- Last exchange ----------------------") 

    # Save the conversation history when the chat ends
    save_conversation_history(conversation_history, filename=conv_file)
    end_time = time.perf_counter()
    taken_time = int((end_time - start_time) * 1000)
    return resultlist, taken_time

def get_history(selected=0):
    files = []
    i = 0

    filenames = sorted(
        os.listdir("./history/"),
        key=lambda f: os.path.getmtime(os.path.join("./history/", f)),
        reverse=True  # newest first
    )
    for filename in filenames:
    # Check if the file is a JSON file
        if filename.endswith('.json'):
            selected_file = i == selected
            #print(i, selected, selected_file)
            files.append({"value": str(i), "text": filename, "selected": selected_file})
            i +=1
    #print(files)
    return files

def get_models(selected=0):
    """Retrieve model list via Ollama REST API."""
    import requests
    models = []
    i = 0
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            for model_info in data.get("models", []):
                model_name = model_info.get("name", "")
                if ":" in model_name:
                    splitmodel = model_name.split(":")
                    if splitmodel[1] == "latest":
                        model = splitmodel[0]
                    else:
                        model = model_name
                else:
                    model = model_name
                selected_file = i == selected
                models.append({"value": str(i), "text": model, "selected": selected_file})
                i += 1
    except Exception as e:
        print(f"Error retrieving models: {e}")
    return models


def get_search_backends(selected_backend="searxng"):
    """Return backend options with availability state for the UI."""
    status = get_backends_status()

    searx_available = status["searxng"]["available"]
    ddgs_available = status["ddgs"]["available"]

    options = [
        {
            "value": "searxng",
            "text": "SearXNG",
            "selected": False,
            "disabled": not searx_available,
        },
        {
            "value": "ddgs",
            "text": "DDGS",
            "selected": False,
            "disabled": not ddgs_available,
        },
    ]

    allowed_values = [opt["value"] for opt in options if not opt["disabled"]]
    if selected_backend in allowed_values:
        resolved_backend = selected_backend
    elif allowed_values:
        resolved_backend = allowed_values[0]
    else:
        # No backend available: keep requested choice to make UI state explicit.
        resolved_backend = selected_backend if selected_backend in ("searxng", "ddgs") else "searxng"

    for opt in options:
        opt["selected"] = opt["value"] == resolved_backend

    return options, resolved_backend


def get_backends_status() -> dict:
    """Return detailed backend availability status for diagnostics and UI alignment."""
    import requests

    searx_health_url = os.environ.get('SEARXNG_HEALTH_URL', SEARXNG_HEALTH_URL_DEFAULT)
    searx_health_timeout = float(os.environ.get('SEARXNG_HEALTH_TIMEOUT', SEARXNG_HEALTH_TIMEOUT_DEFAULT))
    searx_available = False
    searx_reason = "unknown"
    searx_status_code = None

    try:
        response = requests.get(searx_health_url, timeout=searx_health_timeout)
        searx_status_code = response.status_code
        searx_available = response.status_code < 500
        if searx_available:
            searx_reason = "ok"
        else:
            searx_reason = f"HTTP {response.status_code}"
    except requests.Timeout:
        searx_reason = f"timeout_after_{searx_health_timeout}s"
    except requests.RequestException as e:
        searx_reason = str(e)
    except Exception as e:
        searx_reason = str(e)
        searx_available = False

    ddgs_available = False
    ddgs_reason = "unknown"
    try:
        from ddgs_fork import DDGS  # noqa: F401
        ddgs_available = True
        ddgs_reason = "ok"
    except Exception as e:
        ddgs_available = False
        ddgs_reason = f"ddgs_fork_not_installed_or_failed_import: {e}"

    return {
        "searxng": {
            "available": searx_available,
            "reason": searx_reason,
            "health_url": searx_health_url,
            "timeout_seconds": searx_health_timeout,
            "status_code": searx_status_code,
        },
        "ddgs": {
            "available": ddgs_available,
            "reason": ddgs_reason,
        },
    }
    

app = Flask(__name__, template_folder='./')

# Variables globales pour stocker le client et les tools
mcp_tools = None
mcp_client = None

@app.route('/api/conversation/<int:conv_index>', methods=['GET'])
def get_conversation_history(conv_index: int):
    """Return conversation history as JSON (for dynamic loading)."""
    try:
        conv_history = get_history()
        if conv_index < 0 or conv_index >= len(conv_history):
            return {"error": "index_out_of_range"}, 404
        
        conv_file = conv_history[conv_index]['text']
        loaded_history = load_conversation_history(filename=conv_file)
        
        # Format as resultlist for consistency
        resultlist = ["\n\n------------------- Conversation history -------------------"]
        for user_input, bot_response in loaded_history.items():
            resultlist.append(user_input)
            resultlist.append("\nChatbot: " + bot_response)
            resultlist.append("------------------------------------------------------------")
        
        return {"history": resultlist}, 200
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/api/backends/status', methods=['GET'])
def get_backends_status_api():
    """Return detailed backend health status for diagnostics."""
    try:
        selected_backend = request.args.get('selected_backend', DEFAULT_BACKEND)
        options, resolved_backend = get_search_backends(selected_backend=selected_backend)
        status = get_backends_status()
        return {
            "selected_backend": selected_backend,
            "resolved_backend": resolved_backend,
            "status": status,
            "options": options,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/', methods=['GET', 'POST'])
def index():
    global mcp_tools
    conv_history = get_history()
    models = get_models()
    backends, selected_backend = get_search_backends(selected_backend=DEFAULT_BACKEND)
    if request.method == 'POST':
        question = request.form.get('question')
        #print("Question : " + question)
        forget = request.form.get('forget')
        conversation = request.form.get('mySelect')
        model = request.form.get('modelSelect')
        backend_selected = request.form.get('backendSelect', DEFAULT_BACKEND)
        backends, resolved_backend = get_search_backends(selected_backend=backend_selected)
        answer, taken_time = ask(
            question,
            forget,
            models[int(model)],
            conv_history[int(conversation)],
            mcp_tools,
            search_backend=resolved_backend,
        )
        conv_history = get_history(selected=0)
        models = get_models(selected=int(model))
        backends, _ = get_search_backends(selected_backend=resolved_backend)
        #print(type(answer))
        return render_template(
            DEFAULT_TEMPLATE,
            answer=answer,
            calculated_time=taken_time,
            options=conv_history,
            models=models,
            backends=backends,
        )
    else:
        return render_template(DEFAULT_TEMPLATE, options=conv_history, models=models, backends=backends)


if __name__ == "__main__":
    # Initialize agent asynchronously before starting Flask
    async def setup():
        global mcp_tools
        global mcp_client
        mcp_client, mcp_tools = await init_mcp_client()
        print(f"✅ MCP Client initialized with {len(mcp_tools)} tool(s)")
    
    # Run async setup
    asyncio.run(setup())
    
    # Print startup configuration
    print("=" * 60)
    print("🚀 Starting chatLlama3 application")
    print("=" * 60)
    print(f"Default template: {DEFAULT_TEMPLATE}")
    print(f"Default backend: {DEFAULT_BACKEND}")
    print(f"Max iterations agent: {MAX_ITERATIONS}")
    print(f"Port: {APP_PORT}, Host: {APP_HOST}, Debug: {APP_DEBUG}")
    print("=" * 60)
    
    # Start Flask app
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)

