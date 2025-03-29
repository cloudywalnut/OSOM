from flask import Flask, jsonify, request, render_template
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
from flask_socketio import SocketIO
import markdown
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


@app.route('/chat/openai', methods=['POST'])
def chatOpenai():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation"},
            {"role": "user", "content": message}
        ],
        max_tokens=100,
        temperature= 0.7
    )

    return jsonify({"response": markdown.markdown(response.choices[0].message.content)})


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



# Provider of Emotions
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


@app.route('/chat/Grapes', methods=['POST'])
def chatGoogle():
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




if __name__ == "__main__":  
    port = int(os.getenv("PORT", 5000))  # Uses Railway's port, defaults to 5000 if running locally
    socketio.run(app, host="0.0.0.0", port=port, debug=True)