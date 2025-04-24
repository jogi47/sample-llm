from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
import platform
import os

from config import Config

class LLMHandler:
    def __init__(self) -> None:
        # Get model name from configuration
        self.model_name: str = Config.get_model_name()
        
        # Check if we're running on Apple Silicon
        self.is_apple_silicon: bool = platform.system() == "Darwin" and platform.machine() == "arm64"
        
        # Set device appropriately for the platform
        if self.is_apple_silicon:
            # Use MPS (Metal Performance Shaders) on Apple Silicon if available
            self.device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Using device: {self.device} (Apple Silicon: {self.is_apple_silicon})")
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generator: Optional[Any] = None
        
    def load_model(self) -> None:
        """Load the LLM model and tokenizer with memory optimizations"""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Force garbage collection before loading model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Memory optimization configurations
        kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        
        # Platform-specific optimizations
        if self.device == "mps":
            # For Apple Silicon, use float16 with torch_dtype
            kwargs["torch_dtype"] = torch.float16
        elif self.device == "cuda":
            # For CUDA, use float16 as well
            kwargs["torch_dtype"] = torch.float16
        else:
            # For CPU, no additional optimizations - quantization removed due to bitsandbytes incompatibility
            pass
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create text-generation pipeline
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "device_map": "auto",
            "batch_size": 1,
        }
        
        # Create text-generation pipeline
        self.generator = pipeline(
            "text-generation",
            **pipeline_kwargs
        )
        
        # Set memory-saving environment variables
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Helps with MPS memory management
        os.environ["TRANSFORMERS_CACHE"] = "./model_cache"  # Local cache to manage storage
        
        print("Model loaded successfully")
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.95) -> str:
        """Generate a response using the loaded LLM model"""
        if self.model is None or self.tokenizer is None or self.generator is None:
            self.load_model()
        
        # Different prompt templates for different models
        if "TinyLlama" in self.model_name:
            formatted_prompt = f"<human>: {prompt}\n<assistant>:"
        elif "gemma" in self.model_name.lower():
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Generic format for other models
            formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Generate response with conservative settings to limit memory usage
        result = self.generator(
            formatted_prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        generated_text: str = result[0]["generated_text"]
        
        # Extract only the assistant's response based on model-specific format
        if "TinyLlama" in self.model_name and "<assistant>:" in generated_text:
            response = generated_text.split("<assistant>:", 1)[1].strip()
        elif "gemma" in self.model_name.lower() and "<start_of_turn>model" in generated_text:
            response = generated_text.split("<start_of_turn>model\n", 1)[1].strip()
        else:
            # Generic extraction for other models
            assistant_marker = "Assistant:" if "Assistant:" in generated_text else "\nAssistant:"
            if assistant_marker in generated_text:
                response = generated_text.split(assistant_marker, 1)[1].strip()
            else:
                # Fallback extraction
                response = generated_text.replace(formatted_prompt, "").strip()
        
        # Run garbage collection after generation
        gc.collect()
            
        return response

# Create a singleton instance
llm_handler = LLMHandler() 