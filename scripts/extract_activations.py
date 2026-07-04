import torch
from transformers import GPT2Model, GPT2Tokenizer

modelName = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(modelName)
languageModel = GPT2Model.from_pretrained(modelName, torch_dtype=torch.float16)

inputText = "Parallel computing optimizes systems."
tokenizedInput = tokenizer(inputText, return_tensors="pt")

# Obtain layer activations (hidden states) without keeping gradients
with torch.no_grad():
    modelOutputs = languageModel(**tokenizedInput, output_hidden_states=True)

activationTensors = modelOutputs.hidden_states

# Extract activations for the second transformer layer (index 1)
layerIndex = 1
specificLayerActivations = activationTensors[layerIndex]

print("Tensor Shape (Batch, Sequence, Hidden Dimension):")
print(specificLayerActivations.shape)

print("\nRaw FP16 Activation Matrix (First 10 values of the first token):")
print(specificLayerActivations[0, 0, :10])
