"""The deployed entry point.

    uvicorn scripts.serve:app --host 0.0.0.0 --port $PORT

A free instance gives one port, so the API and the page it serves are one app.
This used to mount Gradio here as well; it no longer does. Gradio added 1.6s to
import and ~150MB resident, and on a plan that spins down after 15 minutes both
land on whoever clicks the link next.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.api import app  # noqa: E402,F401
