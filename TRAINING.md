# Model Training Guide

This guide explains how to fine-tune LLM models and push them to Hugging Face Hub for use in your API.

## Prerequisites

- Hugging Face account (username: jogi47)
- Hugging Face API token (get from [HF settings](https://huggingface.co/settings/tokens))
- At least 8GB RAM recommended (though 4GB can work with reduced batch size)

## Quick Start

The easiest way to train a model is to use the provided shell script:

```bash
# Set your Hugging Face token (required to push models to Hub)
export HF_TOKEN=your_huggingface_token_here

# Run the training script with default settings
./train_model.sh
```

This will:
1. Install required dependencies with uv
2. Load a small dataset (DialogSum by default)
3. Fine-tune the TinyLlama-1.1B-Chat-v1.0 model
4. Push the trained model to your Hugging Face account (if HF_TOKEN is set)

## Customizing Training

You can customize training by setting environment variables:

```bash
# Set your Hugging Face token
export HF_TOKEN=your_huggingface_token_here

# Set custom training parameters
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Which base model to fine-tune
export DATASET="knkarthick/dialogsum"              # Which dataset to use
export TEXT_COLUMN="dialogue"                      # Which column in the dataset to use for training
export BATCH_SIZE=4                                # Batch size (smaller = less memory)
export EPOCHS=1                                    # Number of training epochs
export MAX_SAMPLES=2000                            # Maximum samples to use from dataset
export REPO_NAME="my-custom-tinyllama"             # Custom repository name (optional)

# Run the training script
./train_model.sh
```

## Available Datasets (under 50MB)

Here are some small datasets suitable for fine-tuning:

1. **DialogSum** (dialogue summarization):
   - Dataset: `knkarthick/dialogsum`
   - Text Column: `dialogue`
   - Size: ~15MB

2. **SQuAD** (question answering):
   - Dataset: `squad_v2`
   - Text Column: `context`
   - Size: ~40MB

3. **WikiText** (general language modeling):
   - Dataset: `wikitext`
   - Text Column: `text`
   - Size: ~40MB

4. **TinyStories** (simple stories):
   - Dataset: `roneneldan/TinyStories`
   - Text Column: `text`
   - Size: ~40MB

5. **PubMed QA** (medical question answering):
   - Dataset: `pubmed_qa`
   - Text Column: `context`
   - Size: ~5MB

## Training Options

For more control, you can use the `train.py` script directly:

```bash
uv pip run python train.py --help
```

This will show all available options:

```
usage: train.py [-h] [--model_name MODEL_NAME] [--dataset_name DATASET_NAME]
                [--text_column TEXT_COLUMN] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
                [--max_samples MAX_SAMPLES] [--output_dir OUTPUT_DIR]
                [--repository_name REPOSITORY_NAME] [--push_to_hub]
                [--hf_token HF_TOKEN] [--seed SEED]
```

## Using Your Trained Model

After training, the script will output instructions for updating your `config.py` file. 

1. Find the model ID in the output (format: `jogi47/model-name-finetuned-YYYY-MM-DD_HH-MM-SS`)
2. Edit `config.py` and uncomment the "finetuned" section
3. Update the model name and description
4. Set `LLM_MODEL=finetuned` when running your API

Example:

```bash
# Run the API with your fine-tuned model
LLM_MODEL=finetuned ./run_macos.sh
```

## Troubleshooting

### Memory Issues
- Reduce `BATCH_SIZE` to 1 or 2
- Reduce `MAX_SAMPLES` to use less data
- Use a smaller model like `bigscience/bloom-560m`

### Training Too Slow
- Reduce `NUM_EPOCHS` to 1
- Reduce `MAX_SAMPLES` to train on less data
- Use a smaller model like `bigscience/bloom-560m`

### Model Doesn't Push to Hub
- Ensure you've set `export HF_TOKEN=your_token_here`
- Check that your token has write permissions
- Check your internet connection

## Advanced: Parameter-Efficient Fine-Tuning (PEFT)

For more efficient training, especially on limited hardware, the script uses Parameter-Efficient Fine-Tuning techniques automatically when beneficial. 