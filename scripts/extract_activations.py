import os
import torch
from transformers import GPT2Model, GPT2Tokenizer

modelName = "gpt2"
huggingFaceToken = os.environ.get("HF_TOKEN", None)

# Prefer a local copy (run scripts/download_model.py once); fall back to the Hub
repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
localModelDir = os.path.join(repoRoot, "models", modelName)
modelSource = localModelDir if os.path.isdir(localModelDir) else modelName

tokenizer = GPT2Tokenizer.from_pretrained(modelSource, token=huggingFaceToken)
languageModel = GPT2Model.from_pretrained(modelSource, torch_dtype=torch.float16, token=huggingFaceToken)

inputText = "Parallel computing optimizes systems."
tokenizedInput = tokenizer(inputText, return_tensors="pt")

# Obtain layer activations (hidden states) without keeping gradients
with torch.no_grad():
    modelOutputs = languageModel(**tokenizedInput, output_hidden_states=True)

# hidden_states is a tuple of num_layers + 1 tensors:
#   index 0      -> embedding output (token + position embeddings, before any block)
#   index 1..12  -> output of each transformer block (GPT-2 small has 12)
activationTensors = modelOutputs.hidden_states
print(f"Number of hidden states: {len(activationTensors)} (embeddings + 12 blocks)")

layerIndices = [0, 1, 6, 12]
for layerIndex in layerIndices:
    layerActivations = activationTensors[layerIndex]
    layerLabel = "embeddings" if layerIndex == 0 else f"block {layerIndex}"
    print(f"\nLayer {layerIndex} ({layerLabel}) — shape {tuple(layerActivations.shape)}")
    print("First 10 values of the first token:")
    print(layerActivations[0, 0, :10])

# Final output of the model (last block + final layer norm).
# GPT2Model has no LM head, so this is hidden vectors, not token logits.
finalOutput = modelOutputs.last_hidden_state
print(f"\nFinal output (last_hidden_state) — shape {tuple(finalOutput.shape)}")
print("Same tensor as hidden_states[-1]:", torch.equal(finalOutput, activationTensors[-1]))
print(finalOutput[0, 0, :10])

# %%
