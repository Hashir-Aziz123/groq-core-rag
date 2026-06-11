import time
from src.groq_client import GroqClient

def run_api_diagnostic():
    print("[INFO] Initiating Groq API diagnostic sequence...")
    try:
        # Initialize the client (will crash here if API key is missing)
        client = GroqClient()
        print(f"[INFO] Client instantiated. Target model: {client.model}")
        
        # Formulate a minimal test payload
        messages = [
            {"role": "system", "content": "You are a highly precise, zero-fluff diagnostic terminal."},
            {"role": "user", "content": "Define the concept of 'fog of war' in grand strategy games in exactly one sentence."}
        ]
        
        print("[INFO] Firing payload at Groq servers...")
        start_time = time.time()
        
        # Execute the network call
        response = client.generate_content(messages, temperature=0.0)
        
        latency = time.time() - start_time
        print(f"[SUCCESS] Handshake complete. Latency: {latency:.2f} seconds.")
        print("-" * 60)
        print("RESPONSE:")
        print(response)
        print("-" * 60)
        
    except ValueError as ve:
        print(f"\n[CRITICAL] Environment Error: {ve}")
        print("Fix: Ensure GROQ_API_KEY is exported in this exact terminal session.")
    except Exception as e:
        print(f"\n[CRITICAL] Network/API Error: {e}")

if __name__ == "__main__":
    run_api_diagnostic()