"""
OpenEnv multi-mode deployment server entry point.
Wraps the FastAPI app from app.main for `uv run server` compatibility.
"""

import uvicorn
from app.main import app


def main():
    """Entry point for `uv run server`."""
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
