#!/usr/bin/env python
"""
Fine-tune a small LLM model on a Hugging Face dataset and push to HF Hub.
"""
import os
import argparse
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from huggingface_hub import login, HfFolder, create_repo
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default HF username
HF_USERNAME = "jogi47"

class TextDataset(Dataset):
    """Custom dataset for text samples"""
    
    def __init__(self, tokenizer, texts: List[str], max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

def prepare_dataset(dataset_name: str, text_column: str, max_samples: int = 5000) -> List[str]:
    """Load and prepare dataset for training"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load a small dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Get the train split if available
    if "train" in dataset:
        data = dataset["train"]
    else:
        # Use the first available split
        data = dataset[list(dataset.keys())[0]]
    
    # Extract text column
    if text_column not in data.column_names:
        available_columns = ", ".join(data.column_names)
        raise ValueError(f"Column '{text_column}' not found. Available columns: {available_columns}")
    
    # Limit dataset size
    if len(data) > max_samples:
        logger.info(f"Sampling {max_samples} examples from dataset with {len(data)} examples")
        data = data.select(range(max_samples))
    else:
        logger.info(f"Using all {len(data)} examples from dataset")
    
    # Extract text from dataset
    texts = data[text_column]
    
    # Convert to list
    texts = [str(text) for text in texts]
    
    return texts

def train_model(
    model_name: str,
    output_dir: str,
    dataset_name: str,
    text_column: str = "text",
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 1,
    max_samples: int = 5000,
    push_to_hub: bool = True,
    repository_name: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Fine-tune model on dataset and push to HF Hub"""
    
    # Define a unique model ID
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if repository_name:
        hf_model_id = f"{HF_USERNAME}/{repository_name}"
    else:
        model_short_name = model_name.split("/")[-1]
        hf_model_id = f"{HF_USERNAME}/{model_short_name}-finetuned-{timestamp}"
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )
    
    # Get the dataset
    texts = prepare_dataset(dataset_name, text_column, max_samples)
    
    # Create a custom dataset
    dataset = TextDataset(tokenizer, texts, max_length=512)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Login to Hugging Face if pushing to Hub
    if push_to_hub and hf_token:
        logger.info("Logging in to Hugging Face Hub")
        login(token=hf_token, add_to_git_credential=True)
        
        # Create the repository if it doesn't exist
        try:
            create_repo(hf_model_id, private=False, exist_ok=True)
            logger.info(f"Repository {hf_model_id} is ready")
        except Exception as e:
            logger.warning(f"Error creating repository: {e}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.01,
        push_to_hub=push_to_hub,
        hub_model_id=hf_model_id if push_to_hub else None,
        hub_token=hf_token if push_to_hub else None,
        fp16=(device.type == "cuda"),  # Use fp16 on CUDA
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the model locally
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub if requested and not already done by the trainer
    if push_to_hub and not training_args.push_to_hub:
        logger.info(f"Pushing model to Hugging Face Hub: {hf_model_id}")
        try:
            model.push_to_hub(hf_model_id, use_auth_token=hf_token)
            tokenizer.push_to_hub(hf_model_id, use_auth_token=hf_token)
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
    
    logger.info("Training complete!")
    return hf_model_id

def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM model and push to HF Hub")
    
    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Name of the pre-trained model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum", 
                        help="Name of the dataset on Hugging Face Hub")
    parser.add_argument("--text_column", type=str, default="dialogue", 
                        help="Column in the dataset containing the text to train on")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs to train for")
    parser.add_argument("--max_samples", type=int, default=5000, 
                        help="Maximum number of samples to use from the dataset")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./model_output", 
                        help="Directory to save the model to")
    parser.add_argument("--repository_name", type=str, default=None, 
                        help="Name of the repository on Hugging Face Hub")
    
    # Hugging Face arguments
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push the model to Hugging Face Hub")
    parser.add_argument("--hf_token", type=str, default=None, 
                        help="Hugging Face token for pushing to Hub")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Look for HF token in environment variable if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    if args.push_to_hub and not hf_token:
        logger.warning("No Hugging Face token provided. Will not push to Hub.")
        logger.warning("To push to Hub, set the HF_TOKEN environment variable or pass --hf_token")
        args.push_to_hub = False
    
    # Run training
    hf_model_id = train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        repository_name=args.repository_name,
        hf_token=hf_token,
    )
    
    logger.info(f"Model trained and saved to: {hf_model_id}")
    logger.info(f"To use this model in the API, update config.py with:")
    logger.info(f"  \"finetuned\": {{")
    logger.info(f"      \"name\": \"{hf_model_id}\",")
    logger.info(f"      \"description\": \"Your fine-tuned model\",")
    logger.info(f"      \"min_ram_gb\": 8,")
    logger.info(f"      \"recommended_for_macos\": True")
    logger.info(f"  }}")

if __name__ == "__main__":
    main() 