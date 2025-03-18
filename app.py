from flask import Flask, jsonify, request
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

app = Flask(__name__)

client = OpenAI(
  api_key=os.getenv('apiKey')
)


@app.route('/')
def home():
    return 'Server is Running'


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}]
    )

    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=5000, debug=True)
