from flask import Flask, jsonify, request, render_template
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
        messages=[
            {"role": "user", "content": message},
            {"role": "system", "content": "Please act like a friend to a child and use very simple easy tone to teach him the basics of the solar system once he/she starts the conversation"}
        ],
        max_tokens=100,
        temperature= 2
    )

    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":  
    port = int(os.getenv("PORT", 5000))  # Uses Railway's port, defaults to 5000 if running locally
    app.run(host="0.0.0.0", port=port, debug=True)
