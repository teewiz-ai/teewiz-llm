import os
import json

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from openai import OpenAI

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/images/generate")
async def generate_image(request: Request):
    """
    Non-streaming image generation endpoint.
    Expects JSON: { prompt: str, n?: int, size?: str, format?: str }
    Returns JSON with base64-encoded images.
    """
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")

    n = body.get("n", 1)
    size = body.get("size", "1024x1024")

    resp = openai.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=n,
        size=size,
        background=body.get("background", None),
        quality=body.get("quality", None),
    )
    images = []
    for item in resp.data:
        b64 = getattr(item, "b64_json", None)
        if b64:
            images.append(b64)
    return JSONResponse({"images": images})

@app.post("/images/generate/stream")
async def generate_image_stream(request: Request) -> StreamingResponse:
    """
    Streaming image generation endpoint (NDJSON over HTTP).
    Expects JSON: { prompt: str, partial_images?: int }
    Streams JSON lines with each chunk as received.
    """
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")

    stream = openai.responses.create(
        model="gpt-4.1",
        input=prompt,
        tools=[{"type": "image_generation", "partial_images": body.get("partial_images", 0)}],
        stream=True
    )

    def generate():
        for event in stream:
            if event.type == "response.image_generation_call.partial_image":
                b64 = event.partial_image_b64
                print("Yielding")
                yield json.dumps({ "type": "partial", "b64": b64 }) + "\n"
            elif event.type == "response.image_generation_call":
                yield json.dumps({ "type": "final", "b64": event.result }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

# Lambda entry point
handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
