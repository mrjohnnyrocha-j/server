from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_core.messages import HumanMessage
from redis import Redis
from groq import AsyncGroq
import aiofiles
import whisper
from pprint import pprint
import json
import requests
import logging
import io
import hashlib
import os
from dotenv import load_dotenv

env = load_dotenv()
logger = logging.getLogger(__name__)

#from .database import database, settings
from .scripts import CONTEXT, SAFETY_GATE
from .graph import Graph

redis_client = Redis(host='localhost', port=6379, db=0)
whisper_model = whisper#.load_model("base")

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://localhost:3001" "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the j API"}


@app.get("/api/chat")
async def fetch_news(query: str = "hi"):
    """
    Endpoint to fetch news and generate a summary based on a user query.
    Utilizes a stream response to handle potentially long-running AI generation tasks.
    """
    news_items = "results"
    if not news_items or "results" not in news_items:
        raise HTTPException(
            status_code=404, detail="No news items found for the given query"
        )

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Query: {query}\nNews Items: {news_items['results']}",
        },
    ]

    return StreamingResponse(generate_responses(messages), media_type="text/plain")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str
    sentiment_summary: str
    sentiment_score: float


@app.post("/api/chat", response_model=ChatResponse, response_class=HTMLResponse)
async def generate_responses(
    request: ChatRequest,
):
    """
    Generate responses from the LLM in a streaming fashion using the Groq API and return them as HTML.
    """

    try:
        #agent = SentimentAnalysisAgent()

        message = request.message
        print("Message:", message)
        #sentiment_summary, sentiment_score = agent.handle(message)

        inputs = {
            "keys": {
                "question": message,
            }
        }

        graph = Graph()

        app = graph.build()

        outputs = app.invoke(inputs)
        output = outputs['keys']['generation']

        content =  f"Regenerate in a professional and concise manner the answer {output} to the question {message}. Just output the generated answer and nothing else." 
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            model=os.getenv("LOCAL_LLM"),
        )
        response_content = chat_completion.choices[0].message.content

        return HTMLResponse(
            content=f"{response_content}",
            status_code=200,
        )

    except Exception as e:
        print("Error during API call:", e)
        return HTMLResponse(
            content=f"Error processing your message. {e}",
            status_code=500,
        )

async def associate_command(transcription: str) -> str:
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"First, recognise the intent of the user. If he is plainly joking or no intent that can be used by a computer to perform an action in a existant website (such as switching pages, making a call, sending a text, message another user), just output you don't know what to do. Otherwise, use the following transcription to create a typescript function that executes the intent of the user: '{transcription}'",
                }
            ],
            model=os.getenv("LOCAL_LLM"),
        )
        command = chat_completion.choices[0].message.content
        return command
    except Exception as e:
        logger.error(f"Error during command association j API call: {e}")
        return "Unknown command"

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Whisper model.
    """
    try:
        audio_bytes = await file.read()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        cached_result = redis_client.get(audio_hash)

        if cached_result:
            cached_result = json.loads(cached_result.decode('utf-8'))
            return {"transcription": cached_result["transcription"], "audio_key": audio_hash}

        audio_path = f"temp_{audio_hash}.wav"
        async with aiofiles.open(audio_path, 'wb') as out_file:
            await out_file.write(audio_bytes)

        result = whisper_model.transcribe(audio_path)
        os.remove(audio_path)  # Clean up the temporary file

        transcription = result["text"]
        command = await associate_command(transcription)

        # Cache both audio, transcription, and command in Redis
        redis_client.set(audio_hash, json.dumps({"transcription": transcription, "command": command, "audio": audio_bytes.hex()}), ex=3600)  # Cache for 1 hour

        return {"transcription": transcription, "command": command, "audio_key": audio_hash}
    except Exception as e:
        print("Error during transcription:", e)
        raise HTTPException(status_code=500, detail=f"Error processing your audio file: {e}")

# Endpoint to retrieve cached audio
@app.get("/api/audio/{audio_key}")
async def get_audio(audio_key: str):
    """
    Retrieve cached audio file from Redis.
    """
    cached_result = redis_client.get(audio_key)
    if not cached_result:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    cached_result = json.loads(cached_result.decode('utf-8'))
    audio_bytes = bytes.fromhex(cached_result["audio"])
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

# @app.on_event("startup")
# async def startup_db_client():
#     client = AsyncIOMotorClient(settings.MONGODB_URI)
#     database = client[settings.DATABASE_NAME]

# @app.on_event("shutdown")
# async def shutdown_db_client():
#     database.client.close()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI with MongoDB"}

# # Example route using the database
# @app.post("/users/")
# async def create_user(user: dict):
#     user_collection = database.get_collection("users")
#     result = await user_collection.insert_one(user)
#     return {"id": str(result.inserted_id)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
