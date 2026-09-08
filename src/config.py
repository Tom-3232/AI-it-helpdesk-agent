import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "llama3.2"

def check_ollama_status():
    """
    Checks if Ollama service is reachable and whether the target model (llama3.2) is pulled.
    Returns dict: {'online': bool, 'has_model': bool, 'message': str, 'models': list}
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check if llama3.2 or llama3.2:latest is in model list
            has_target = any(DEFAULT_MODEL in m for m in models)
            if has_target:
                return {
                    "online": True,
                    "has_model": True,
                    "message": f"Ollama is online with model '{DEFAULT_MODEL}'.",
                    "models": models
                }
            else:
                return {
                    "online": True,
                    "has_model": False,
                    "message": f"Ollama is running, but model '{DEFAULT_MODEL}' is missing. Run: ollama pull {DEFAULT_MODEL}",
                    "models": models
                }
    except Exception as e:
        return {
            "online": False,
            "has_model": False,
            "message": f"Ollama is not running. Please install Ollama and run: ollama pull {DEFAULT_MODEL}",
            "models": []
        }
