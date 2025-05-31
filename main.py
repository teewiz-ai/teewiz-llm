import os, base64, mimetypes
import requests
import json
import io

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


def _inline_image_from_url(url: str) -> dict:
    """Download URL → data URI block for the Responses API."""
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        raise HTTPException(502, f"Failed to download {url}: {e}")
    if r.status_code != 200:
        raise HTTPException(502, f"GET {url} returned {r.status_code}")

    mime = r.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "image/png"
    b64 = base64.b64encode(r.content).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{b64}"
    }

def _download_image_as_fileobj(url: str) -> io.BytesIO:
    """
    Download the image at `url` into an in-memory BytesIO with a `.name` attribute.
    Raises HTTPException on failure.
    """
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        raise HTTPException(502, f"Failed to download image from {url}: {e}")

    if resp.status_code != 200:
        raise HTTPException(502, f"GET {url} returned status {resp.status_code}")

    # Guess extension from Content-Type or URL
    mime = resp.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "image/png"
    ext = mimetypes.guess_extension(mime) or ".png"
    filename = url.split("/")[-1].split("?")[0] or f"input{ext}"

    file_bytes = resp.content
    file_obj = io.BytesIO(file_bytes)
    file_obj.name = filename  # openai expects `file.name`
    return file_obj

@app.post("/images/generate")
async def generate_image(request: Request):
    """
    Non-streaming image generation/edit endpoint.
    Expects JSON: {
      prompt: str,
      n?: int,
      size?: str,
      quality?: str,
      background?: str,
      sampleImageUrl?: string | [string, ...]
    }

    If `sampleImageUrl` is provided, we download each URL and call openai.images.edit(...)
    with model="gpt-image-1", passing the list of file-like objects plus the prompt.

    Otherwise, we fall back to openai.images.generate(...) as before.
    Returns JSON {"images": [<base64-string>, ...]}.
    """
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(400, "Missing 'prompt'")

    sample = body.get("sampleImageUrl")

    # ----------- Branch A: sampleImageUrl provided → use images.edit() -----------
    if sample:
        # Normalize to list of URLs
        urls = [sample] if isinstance(sample, str) else sample
        if not (isinstance(urls, list) and all(isinstance(u, str) for u in urls)):
            raise HTTPException(400, "`sampleImageUrl` must be a string or list of strings")

        # Download each URL into a BytesIO file-like object
        file_objs = []
        for url in urls:
            file_obj = _download_image_as_fileobj(url)
            file_objs.append(file_obj)

        # Call the Image Edit endpoint with all reference images
        try:
            resp = openai.images.edit(
                model="gpt-image-1",
                image=file_objs,     # list of file-like objects
                prompt=prompt,
            )
        except Exception as e:
            raise HTTPException(502, f"OpenAI Images Edit API error: {e}")

        # Extract base64 outputs
        images = []
        for item in getattr(resp, "data", []):
            b64 = getattr(item, "b64_json", None)
            if b64:
                images.append(b64)

        if not images:
            raise HTTPException(500, "OpenAI Images Edit API returned no images")
        return JSONResponse({"images": images})

    # ----------- Branch B: no sampleImageUrl → use images.generate() -----------
    try:
        resp = openai.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=body.get("n", 1),
            size=body.get("size", "1024x1024"),
            quality=body.get("quality"),
            background=body.get("background"),
            moderation="low"
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI Images API error: {e}")

    images = []
    for item in getattr(resp, "data", []):
        b64 = getattr(item, "b64_json", None)
        if b64:
            images.append(b64)

    if not images:
        raise HTTPException(500, "OpenAI Images API returned no data")
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
