import requests
import json
from typing import Dict, Any

def test_llm_endpoint() -> None:
    """Test the LLM endpoint with a sample prompt"""
    url: str = "http://localhost:8000/llm/generate"
    
    # Sample prompt
    payload: Dict[str, Any] = {
        "prompt": "What is the capital of France?",
        "max_length": 200
    }
    
    # Send POST request to the endpoint
    print(f"Sending request to {url} with prompt: {payload['prompt']}")
    
    try:
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nLLM Response:")
            print("-" * 50)
            print(result["response"])
            print("-" * 50)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm_endpoint() 