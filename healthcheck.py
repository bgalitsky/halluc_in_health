from fastapi import FastAPI
import socket
#uvicorn healthcheck:app --host 0.0.0.0 --port 9000
#http://54.82.56.2:9000/health


app = FastAPI()

# List of services and their ports
SERVICES = {
    "discourse_parser": 8501,
    "fastapi_parser": 8000,
    "logic_runner": 5000,
    "rule_attenuation": 8502,
    "diagnosis_verifier": 8503,
}

def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if host:port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

@app.get("/health")
def health():
    results = {}
    for name, port in SERVICES.items():
        results[name] = "online" if check_port("127.0.0.1", port) else "offline"
    return results
