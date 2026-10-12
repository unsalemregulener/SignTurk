"""Standalone dev/test server for the text_processing pipeline.

Lets you test the text/LLM pipeline without the heavy ML/CV stack (no
tensorflow / mediapipe / opencv / torch and no webcam):

    pip install fastapi uvicorn pydantic
    python -m text_processing.devserver
    # or: uvicorn text_processing.devserver:app --reload --port 8000
    # then open  http://127.0.0.1:8000/docs

Optional extras (only if you want to exercise those paths):
    pip install gtts                 # audio (synthesize_audio=true)
    pip install huggingface_hub      # + export HF_TOKEN  -> test cloud ML (Qwen)

It mounts the real ``/api/text/*`` routers, without the authentication the
main application adds, so the endpoints can be tried directly from the
interactive OpenAPI page.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from text_processing.web.router import admin_router, router

app = FastAPI(title="SignTurk text processing API")
app.include_router(router)
app.include_router(admin_router)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        "<h1>SignTurk text processing API</h1>"
        '<p>The grammar service is running. <a href="/docs">Open the interactive API documentation</a>.</p>'
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  SignTurk text processing API -> http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
