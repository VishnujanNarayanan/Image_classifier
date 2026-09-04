"""Hugging Face Space entry point.

The Space runs as a single process, so there is no separate API to call: API_URL
stays unset and scripts/demo.py loads the model in-process. This file exists only
because Spaces looks for `app.py` at the root.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.pop("API_URL", None)          # in-process; there is no sidecar service
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "7860")    # the port Spaces expects
os.environ.setdefault("MODEL_PATH", os.path.join(HERE, "artifacts", "deep.keras"))

from scripts.demo import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
