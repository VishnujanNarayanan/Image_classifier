"""One ASGI app serving both the API and the UI.

A free Render instance gives one port and 512MB. Running uvicorn and Gradio as
separate processes -- which is what the Dockerfile does locally -- would need two
ports and would load the model twice. Mounting the UI onto the API gives
POST /predict and the demo page on the same port, sharing one ONNX session.

    uvicorn scripts.serve:app --host 0.0.0.0 --port $PORT
"""
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.api import app as api_app  # noqa: E402
from scripts.demo import demo  # noqa: E402

# The UI at /, the API beneath it. Gradio keeps its own routes under /gradio_api,
# so POST /predict and GET /health are unaffected.
app = gr.mount_gradio_app(api_app, demo, path="/")
