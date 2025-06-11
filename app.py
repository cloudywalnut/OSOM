from flask import Flask, jsonify, request, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
from flask_socketio import SocketIO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import markdown

import edge_tts
import asyncio

load_dotenv()
import os

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
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




####### Langchain Endpoints ########

model = ChatOpenAI(model = "gpt-4o-mini")

genz_slangs = ["lit", "bet", "fam", "no cap", "yeet", "vibe check", "sus", "flex"]

prompt_template_normal = ChatPromptTemplate(
    [
        ("system", 
         "You are a friendly teacher answering a student's question about the {topic} in a very simple, easy tone. "
         "Keep the explanation fun, clear, and age-appropriate. Always respond warmly, encourage curiosity and strictly never go off topic"
         ""),
        ("human", "{message}")
    ]
)

prompt_template_genz = ChatPromptTemplate(
    [
        ("system", 
         "You are a friendly teacher answering a student's question about the {topic} in a very simple, easy tone. "
         "Sprinkle in some Gen Z and Gen Alpha slang from this list when appropriate: {genz_slangs}."  
         "Make the chat fun and relatable, but keep explanations clear and simple and strictly never go off topic."
         "Never use emojis or greetings (unless user greets)"),
        ("human", "{message}")
    ]
)


@app.route('/chat', methods = ['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    topic = data.get("topic", "")
    mode_genz = data.get("mode_genz", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    if mode_genz:
        chain = prompt_template_genz | model 
    else:
        chain = prompt_template_normal | model         
    response = chain.invoke({"topic": topic, "genz_slangs": genz_slangs, "message": message, })
    return jsonify({"response": response.content})


@app.route('/chat/agentic', methods = ['POST'])
def chat_agentic():
    data = request.json
    tools = data.get("tools","")
    message = [HumanMessage(data.get("message", ""))]
    # The message passed in invoke should either be a string or a list, so can't directly use HumanMessage
    # without having it in a list
    response = model.invoke(message, tools = tools)
    if response.tool_calls:
        return jsonify({"response": response.tool_calls})
    else:
        return jsonify({"response": response.content})

@app.route('/eq-question', methods=['POST'])
def eq_question():
    data = request.json

    # Extract the details from the request
    age = data.get('age')
    gender = data.get('gender')
    interests = data.get('interests', [])
    last_topic = data.get('last_topic', '')
    current_topic = data.get('current_topic', '')
    prev_questions = data.get('prev_questions', [])
    prev_responses = data.get('prev_responses', [])
    last = data.get("last", False)

    # Deals with last question differently to close the convo
    if not last:
        system_instruction = (
            "You are a fun and emotionally intelligent AI that engages kids in simple EQ conversations. "
            "Keep things creative, playful, and age-appropriate. Always stay strictly on the current topic. "
            "If no previous response exists, ask an engaging, imaginative question to start the conversation."
        )

        user_content = (
            f"Age: {age}\n"
            f"Gender: {gender}\n"
            f"Interests: {interests}\n"
            f"Last Topic: {last_topic}\n"
            f"Current Topic: {current_topic}\n"
            f"Previous Questions: {prev_questions}\n"
            f"Previous Responses: {prev_responses}\n"
            f"Instruction: Based on the above, respond creatively to the child’s last message if it exists. "
            f"Then ask a short, fun EQ question related only to the current topic. Avoid greetings and keep it simple."
        )

    else:
        system_instruction = (
            "You are a warm and encouraging AI that concludes EQ conversations with kids. "
            "Create a positive, uplifting closing message that wraps up the topic nicely. "
            "Include a fun or helpful tip related to the current topic to keep the child interested."
        )

        user_content = (
            f"Age: {age}\n"
            f"Gender: {gender}\n"
            f"Interests: {interests}\n"
            f"Last Topic: {last_topic}\n"
            f"Current Topic: {current_topic}\n"
            f"Previous Questions: {prev_questions}\n"
            f"Previous Responses: {prev_responses}\n"
            f"Instruction: Based on the above, respond with a short, positive closing statement that concludes the conversation. "
            f"Make it uplifting and include a unique tip or fun fact about the current topic. End with see you in the next session"
        )

    message = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]

    response = model.invoke(message)
    ai_question = response.content.strip()
    return jsonify({"question": ai_question})



# Question and Answer, class understanding:
# Changes Needed, Questions will be given fine tune per user persona and change the evaluation to give the score as well
class Questions(BaseModel):
    questions: list[str] = Field(description="The question, to be asked")    

@app.route('/class_questions', methods=['POST'])
def class_questions():    
    data = request.json
    topic = data.get("topic", "")
    interests = data.get("interests", [])
    num_questions = data.get("num_questions", 5)

    prompt_template = ChatPromptTemplate([
            ("system", 
            "Based on the kids interests: {interests} come up with {num_questions} unique short answer questions on {topic} that"
            "require critical thinking and not memorizing facts. The questions should be unique, short and easy to understand"
            )
        ]
    )

    structured_model = model.with_structured_output(Questions)
    chain = prompt_template | structured_model
    response = chain.invoke({"topic": topic, "interests": interests, "num_questions": num_questions})
    return jsonify({"questions": response.questions})


@app.route('/class_questions_v2', methods=['POST'])
def class_questions_v2():    
    data = request.json
    questions = data.get("questions", [])
    interests = data.get("interests", [])

    prompt_template = ChatPromptTemplate([
            ("system", 
            "Based on the kids interests: {interests} transform these questions: {questions}, making them unique short and easy"
            "for the student/kid to understand"
            )
        ]
    )

    structured_model = model.with_structured_output(Questions)
    chain = prompt_template | structured_model
    response = chain.invoke({"questions": questions, "interests": interests})
    return jsonify({"questions": response.questions})


# Make a v3 question generator as well that would work on the key class understanding points


@app.route('/evaluate_questions', methods = ['POST'])
def evaluate_responses():
    data = request.json
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    
    # Need to improve on this prompt
    prompt_template = ChatPromptTemplate([
            ("system", 
            "Evaluate the students answers {answers} against these questions: {questions}. Appreciate them and "
            "Provide constructive feedback ending on a positive note with the hope of seeing them again in other classes and appreciating them."
            "Keep it succint, short and easy to understand"
            )
        ]
    )
    
    chain = prompt_template | model
    response = chain.invoke({"questions": questions, "answers": answers})
    return jsonify({"response": response.content})


# Was Facing some issues in deployment after using langchain as some versions were conflicting, hence i deleted the whole langchain
# and only the langchain packages actually used were kept, I updated the requirements again and then it ran.
if __name__ == "__main__":  
    port = int(os.getenv("PORT", 5000))  # Uses Railway's port, defaults to 5000 if running locally
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
