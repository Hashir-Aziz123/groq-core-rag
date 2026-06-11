import os
from dotenv import load_dotenv
import requests

class GroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        load_dotenv()

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_content(self, messages: list, temperature=0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]