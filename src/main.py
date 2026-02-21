from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import the chain creation function
from src.chatbot import create_chatbot_chain

# Initialize the FastAPI app
app = FastAPI()

# Load the model and create the chain ONCE when the server starts
logger.info("Starting server and loading models...")
try:
    # Now returns both the chain and the retriever instance
    chatbot_chain, retriever = create_chatbot_chain()
    logger.info("Model and database loaded successfully.")
except Exception as e:
    logger.critical("Failed to load model or database: %s", e)
    raise e


# Define the request model: what the app will send to the server
class Question(BaseModel):
    text: str
    mode: str = "conversation"  # Options: "conversation", "data"
    audience: str = "patient"  # Options: "patient", "pharmacist"
    language: str = "English"  # e.g., "English", "Indonesian", "Spanish"
    verbosity: str = "detailed"  # Options: "concise", "detailed"
    format: str = "text"  # Options: "text", "json"


@app.get("/")
async def read_root():
    logger.info("Health check endpoint called.")
    return {"message": "Drug Interaction Chatbot API is running"}


# This is your main API endpoint
@app.post("/ask")
async def ask_question(question: Question):
    """
    Receives a question and a mode.
    - mode="conversation": Returns LLM-generated response (default).
    - mode="data": Returns raw retrieved interaction info from the seed CSV.
    """
    logger.info(
        "Received /ask request: %s... | Mode: %s", question.text[:50], question.mode
    )

    if question.mode == "data":
        # Direct retrieval without LLM
        # Use the retriever instance method
        answer = retriever.retrieve(question.text)
        logger.info("Returning data mode response.")
        return {"answer": answer}

    # Default: Conversation mode (LLM)
    # Use ainvoke for asynchronous execution if supported by the chain components
    # Otherwise, FastAPI handles sync defs in a threadpool, but async def + sync call blocks the loop.
    # Since we are using llama.cpp which might be CPU bound, ainvoke is preferred if implemented,
    # or we keep it sync def to let FastAPI thread it.
    # However, LangChain's ainvoke usually handles threading for sync runnables.
    try:
        answer = await chatbot_chain.ainvoke(
            {
                "question": question.text,
                "audience": question.audience,
                "language": question.language,
                "verbosity": question.verbosity,
                "format": question.format,
            }
        )
        logger.info("Generated LLM response successfully.")
        return {"answer": answer}
    except Exception as e:
        logger.error("Error generating response: %s", e)
        return {
            "answer": "I'm sorry, I encountered an error while processing your request."
        }


@app.post("/stream")
async def stream_question(question: Question):
    """
    Streams the answer token-by-token using Server-Sent Events (SSE).
    Useful for real-time UI updates.
    """
    logger.info("Received /stream request: %s...", question.text[:50])

    if question.mode == "data":
        # Data mode is fast enough, no need to stream, but we wrap it in a generator for consistency
        async def data_generator():
            # Use the retriever instance method
            answer = retriever.retrieve(question.text)
            yield answer

        return StreamingResponse(data_generator(), media_type="text/plain")

    # Conversation mode: Stream from LLM
    async def response_generator():
        try:
            # 1. First, search for the data and send it as a JSON event
            interaction_data = retriever.search(question.text)

            # If valid data found (not an error dictionary), send it
            if interaction_data and "error" not in interaction_data:
                # We serialize the dictionary to JSON string
                json_data = json.dumps(interaction_data)
                # Send as a special event or just a line with a prefix
                # Using standard SSE format with an 'event' type if client supports it,
                # but 'uvicorn' StreamingResponse with media_type="text/event-stream" usually just sends data: ...
                # We will send a special JSON line first.
                yield f"{json_data}\n\n"

            # 2. Then stream the LLM response chunks
            async for chunk in chatbot_chain.astream(
                {
                    "question": question.text,
                    "audience": question.audience,
                    "language": question.language,
                    "verbosity": question.verbosity,
                    "format": question.format,
                }
            ):
                # Check if chunk is a string or has content
                if isinstance(chunk, str):
                    yield chunk
                elif hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)

                # Optional: small delay to ensure smooth streaming if local inference is too bursty
                # await asyncio.sleep(0.01)
            logger.info("Streaming completed successfully.")
        except Exception as e:
            logger.error("Error during streaming: %s", e)
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(response_generator(), media_type="text/event-stream")
