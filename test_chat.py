#!/usr/bin/env python
"""
Test script for the chat interface API
"""
import requests
import json
from typing import Dict, Any, List, Optional
import colorama
from colorama import Fore, Style
import argparse

# Initialize colorama
colorama.init()

# Base URL for API
BASE_URL = "http://localhost:8000"

def print_message(role: str, message: str) -> None:
    """Print a colored message based on role"""
    if role == "user":
        print(f"{Fore.GREEN}You: {Style.RESET_ALL}{message}")
    else:
        print(f"{Fore.BLUE}AI: {Style.RESET_ALL}{message}")
    print()

def test_chat_api(model: Optional[str] = None) -> None:
    """Test the chat API with a simple conversation"""
    session = requests.Session()
    
    # Check available models first
    try:
        response = session.get(f"{BASE_URL}/llm/info")
        if response.status_code == 200:
            models_info = response.json()
            print(f"{Fore.YELLOW}Available models:{Style.RESET_ALL}")
            for model_key, model_info in models_info["available_models"].items():
                print(f"- {model_key}: {model_info['description']}")
            print(f"Current model: {models_info['current_model']}")
            print()
        else:
            print(f"Error getting model info: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return
    
    # Set model if specified
    if model:
        try:
            response = session.post(
                f"{BASE_URL}/api/set-model",
                json={"model": model}
            )
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    print(f"{Fore.GREEN}Model set to {model}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Error setting model: {result['message']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Error setting model: {response.status_code} - {response.text}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error setting model: {e}{Style.RESET_ALL}")
    
    # Sample conversation
    conversation: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "What model are you running?"},
        {"role": "user", "content": "Tell me a short joke."}
    ]
    
    # Process each message
    for message in conversation:
        prompt = message["content"]
        print_message("user", prompt)
        
        try:
            # Send to API
            payload = {
                "prompt": prompt,
                "model": model,  # This will be None if not specified
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.95
            }
            
            response = session.post(
                f"{BASE_URL}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print_message("assistant", result["response"])
                print(f"{Fore.YELLOW}Model used: {result['model']}{Style.RESET_ALL}")
                print("-" * 50)
                print()
            else:
                print(f"{Fore.RED}Error: {response.status_code} - {response.text}{Style.RESET_ALL}")
                print("-" * 50)
                print()
        except Exception as e:
            print(f"{Fore.RED}Error sending message: {e}{Style.RESET_ALL}")
            print("-" * 50)
            print()

def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test the chat interface API")
    parser.add_argument("--model", type=str, default=None, 
                       help="Model to use (tiny, small, medium)")
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}Testing Chat API{Style.RESET_ALL}")
    print(f"{Fore.CYAN}=============={Style.RESET_ALL}")
    print()
    
    test_chat_api(args.model)

if __name__ == "__main__":
    main() 