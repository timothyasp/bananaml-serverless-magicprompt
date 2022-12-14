# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from transformers import pipeline, set_seed

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

if __name__ == "__main__":
    download_model()
