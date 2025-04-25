from typing import Dict, Any
import os
import platform

class Config:
    """Configuration class to control LLM behavior"""
    
    # Set this to True if running on a machine with less than 8GB RAM
    LOW_MEMORY_MODE: bool = False
    
    # Detect Apple Silicon
    IS_APPLE_SILICON: bool = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    # Available models, ordered from smallest to largest
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "description": "Small 1.1B parameter model, works on systems with limited RAM (4-8GB)",
            "min_ram_gb": 4,
            "recommended_for_macos": True
        },
        "small": {
            "name": "bigscience/bloom-560m",
            "description": "560M parameter model, good balance of size and capability for macOS",
            "min_ram_gb": 4,
            "recommended_for_macos": True
        },
        "medium": {
            "name": "microsoft/phi-2",
            "description": "2.7B parameter model, good capabilities while still working on 8GB RAM",
            "min_ram_gb": 8,
            "recommended_for_macos": True
        },
        # Add this section after training a model with train.py
        # "finetuned": {
        #     "name": "jogi47/TinyLlama-1.1B-Chat-v1.0-finetuned-YYYY-MM-DD_HH-MM-SS",
        #     "description": "Custom fine-tuned model",
        #     "min_ram_gb": 8,
        #     "recommended_for_macos": True
        # }
    }
    
    # Default model to use - can be overridden with environment variable
    DEFAULT_MODEL: str = "tiny"
    
    # Model to use when on Apple Silicon by default (can be overridden)
    DEFAULT_MACOS_MODEL: str = "tiny"
    
    @classmethod
    def get_model_name(cls) -> str:
        """Get the model name based on configuration and environment"""
        # Check environment variable for model override
        model_key = os.environ.get("LLM_MODEL", 
                                 cls.DEFAULT_MACOS_MODEL if cls.IS_APPLE_SILICON else cls.DEFAULT_MODEL)
        
        # Fallback to tiny model if in low memory mode
        if cls.LOW_MEMORY_MODE and model_key != "tiny":
            print("WARNING: Using 'tiny' model due to LOW_MEMORY_MODE setting")
            model_key = "tiny"
        
        # Fallback to default if specified model isn't available
        if model_key not in cls.AVAILABLE_MODELS:
            print(f"WARNING: Model '{model_key}' not found, using default")
            model_key = cls.DEFAULT_MACOS_MODEL if cls.IS_APPLE_SILICON else cls.DEFAULT_MODEL
        
        model_name = cls.AVAILABLE_MODELS[model_key]["name"]
        print(f"Selected model: {model_key} ({model_name})")
        
        return model_name 