import random
from transformers import pipeline, set_seed
import torch
# adopted from https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion/blob/main/app.py

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    seed = random.randint(100, 1000000)
    set_seed(seed)

    # Run the model
    result = model(prompt, max_length=(len(prompt) + random.randint(60, 90)), num_return_sequences=4)

    # Return the results as a dictionary
    return result
