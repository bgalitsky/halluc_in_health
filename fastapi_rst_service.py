# fastapi_rst_service.py

from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
import uvicorn
from typing import Dict, Any


# ----------------------------
# Define input schema
# ----------------------------
class TextInput(BaseModel):
    text: str


# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(title="RST Discourse Parser API")


# ----------------------------
# Caching wrapper for analyze_rst1
# ----------------------------
@lru_cache(maxsize=256)
def analyze_rst1_cached(text: str) -> Dict[str, Any]:
    """
    Analyze text with RST discourse parser.
    This is cached to prevent repeated calls to ChatGPT for the same text.
    """
    # Replace this with your actual RST parser / ChatGPT call
    result = analyze_rst1(text)
    return result


# ----------------------------
# Example RST function
# ----------------------------
def analyze_rst1(text: str) -> Dict[str, Any]:
    """
    Example placeholder RST analysis.
    In real code, call your ChatGPT / RST parser here.
    """
    # Dummy RST tree
    tree = {
        "text": text,
        "root": {
            "relation": "root",
            "nucleus": text,
            "satellites": []
        }
    }

    # Example: find satellite phrases (just dummy split by commas)
    satellites = [s.strip() for s in text.split(",")[1:]]
    tree["root"]["satellites"] = satellites

    return {
        "tree": tree,
        "satellites_only_with_nucleus": satellites
    }


# ----------------------------
# POST endpoint
# ----------------------------
@app.post("/analyze")
def analyze(input: TextInput):
    """
    Accepts JSON: {"text": "Your text here"}
    Returns: JSON with RST tree and satellites info
    """
    result = analyze_rst1_cached(input.text)
    return result


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)