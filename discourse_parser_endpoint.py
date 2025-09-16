

from discourse_parser_app import analyze_rst1

#nohup python3 fastapi_rst_service.py > fastapi.log 2>&1 &
#http://54.82.56.2:8000/docs#/default/analyze_analyze_post
#

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
#nohup python3 fastapi_rst_service.py > fastapi.log 2>&1 &

#curl -X POST http://<EC2_PUBLIC_IP>:8000/analyze \
#     -H "Content-Type: application/json" \
#     -d '{"text":"Developed by Peking University, the toolkit segments EDUs."}'

#ssh -i C:\Users\User\.ssh\aws_ec2_key.pem ec2-user@54.82.56.2

# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
