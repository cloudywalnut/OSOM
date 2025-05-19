from flask import Flask, jsonify, request, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
from flask_socketio import SocketIO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import markdown

import edge_tts
import asyncio

load_dotenv()
import os

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
  api_key=os.getenv('apiKey')
)

clientGoogle = OpenAI(
    api_key = os.getenv('apiKey-google'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


@app.route('/')
def home():
    return 'Server is Running'

# OpenAI Endpoint
@app.route('/chat/openai', methods=['POST'])
def chatOpenai():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation. Strictly always stick to the topic, do not use emojis, and keep the response short and easy"},
            {"role": "user", "content": message}
        ],
        temperature= 0.7
    )

    return jsonify({"response": response.choices[0].message.content.strip()})

# Google Endpoint
@app.route('/chat/google', methods=['POST'])
def chatGoogle():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = clientGoogle.chat.completions.create(
        model="gemini-2.0-flash-lite",
        n=1,
        messages=[
            {"role": "system", "content": "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation"},
            {"role": "user", "content": message}
        ],
        max_tokens=100,
        temperature= 0.7
    )

    return jsonify({"response": markdown.markdown(response.choices[0].message.content)})

# Websocket also passes the emotions
@socketio.on('message')
def message_emotion(message):

    response_message = clientGoogle.chat.completions.create(
        model="gemini-2.0-flash-lite",
        n=1,
        messages=[
            {"role": "system", "content": "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation, Don't start with a greeting"},
            {"role": "user", "content": message}
        ],
        max_tokens=100,
        temperature= 0.7
    )

    socketio.emit("response", markdown.markdown(response_message.choices[0].message.content))

    response_emotion = clientGoogle.chat.completions.create(
        model="gemini-2.0-flash-lite",
        n=1,
        messages=[
            {"role": "system", "content": "Understand the emotion of the message and return exactly just one word from: Happy, Sad, Angry, Neutral, Excited."},
            {"role": "user", "content": response_message.choices[0].message.content[:20]}
        ],
        max_tokens=2,
        temperature=0.7
    )

    socketio.emit("emotion", markdown.markdown(response_emotion.choices[0].message.content))

# Test endpoint
@app.route('/chat/Grapes', methods=['POST'])
def chatGrapes():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = clientGoogle.chat.completions.create(
        model="gemini-2.0-flash-lite",
        n=1,
        messages=[
            {"role": "system", "content": ("Exclusively assists users with smart home solutions in "
            "Malaysia, representing Grapes Smart Tech, a Loxone Gold Partner. It provides expert "
            "guidance on Loxone Smart Home automation, covering energy efficiency, security, lighting,"
            "climate control, and seamless home integration. Strictly never deviate from topic and"
            "make response short and concise")
            },
            {"role": "user", "content": message}
        ],
        temperature= 0.7
    )

    return jsonify({"response": markdown.markdown(response.choices[0].message.content)})

# Test endpoint
@app.route('/chatApp', methods=['POST'])
def chatApp():
    data = request.json
    message = data.get("message", "")
    system_message = data.get("system_message","")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        n=1,
        messages=[
            {"role": "system", "content": "You need to reply on my behalf to the chat message recieved, as if I am replying to that person in the specified tone or instruction provided"},
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        temperature= 0.7
    )

    return jsonify({"response": response.choices[0].message.content})

# Text to Speach Usage
async def generate_speech(text):
    tts = edge_tts.Communicate(text,  voice="en-US-AvaMultilingualNeural", rate="-20%", pitch="+40Hz")
    output = b""
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            output += chunk["data"]  # Collect all audio data
    return output

def ai_response(message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You must always respond with a concise and playful reply—like a teacher explaining"
                "to a kid. Keep it short and engaging. The conversation should always be directed to"
                "the Solar System and never use emojis")
            },
            {"role": "user", "content": message}
        ],
        temperature= 1
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


@app.route("/tts", methods = ["POST"])
def get_audio():
    data = request.json
    message = data.get("message", "")
    audio = asyncio.run(generate_speech(ai_response(message)))  # Run async function synchronously
    return Response(audio, content_type="audio/mpeg")

# Langchain Endpoints
model = ChatOpenAI(model = "gpt-4o-mini")
prompt_template_normal = ChatPromptTemplate(
    [
        ("system", "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation"),
        ("human", "{message}")
    ]
)
prompt_template_genz = ChatPromptTemplate(
    [
        ("system", "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation"),
        ("system", "occasionally include some Genz and Gen alpha slangs on between"),
        ("human", "{message}")
    ]
)

@app.route('/chat', methods = ['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    mode_genz = data.get("mode_genz", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    if mode_genz:
        chain = prompt_template_genz | model 
    else:
        chain = prompt_template_normal | model         
    response = chain.invoke({"message": message})
    return jsonify({"response": response.content})


@app.route('/chat/agentic', methods = ['POST'])
def chat_agentic():
    data = request.json
    tools = data.get("tools","")
    message = [HumanMessage(data.get("message", ""))]
    # The message passed in invoke should either be a string or a list, so can't directly use HumanMessage
    # without having it in a list
    response = model.invoke(message, tools = tools)
    return jsonify(response.tool_calls[0])

if __name__ == "__main__":  
    port = int(os.getenv("PORT", 5000))  # Uses Railway's port, defaults to 5000 if running locally
    socketio.run(app, host="0.0.0.0", port=port, debug=True)