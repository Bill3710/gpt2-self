# %%
import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
import csv
from itertools import islice

data = []
df = pd.read_csv('./data/raw/fltrace_out/fluidanimate/500_300/fltrace-data-faults-26866-1.out', index_col=1)
data = df['addr'].values.tolist()
# print(df.columns)
# print(df.describe())
print(df.head())
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
validation_data = data[split_index:]
data[0:5]

# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# %%
class MemoryAccessDataset(Dataset):
    def __init__(self, tokenizer, data, sequence_length=10):
        self.input_ids = []
        self.label_ids = []
        self.attention_masks = []
        sequences = ""
        label_sequences = ""

        for index in range(len(data) - 1):
            sequences += f"{data[index]}   "
            label_sequences += f"{data[index + 1]}   "

            if (index + 1) % sequence_length == 0 or (index + 1) == len(data) - 1:
                # print(sequences)
                # print(label_sequences)

                encoding_in = tokenizer(sequences, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
                encoding_label = tokenizer(label_sequences, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)

                self.input_ids.append(encoding_in['input_ids'])
                self.label_ids.append(encoding_label['input_ids'])
                self.attention_masks.append(encoding_in['attention_mask'])

                sequences = ""
                label_sequences = ""


        if sequences:

            encoding_in = tokenizer(sequences, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
            encoding_label = tokenizer(label_sequences, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
            
            self.input_ids.append(encoding_in['input_ids'])
            self.label_ids.append(encoding_label['input_ids'])
            self.attention_masks.append(encoding_in['attention_mask'])
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx].squeeze(), self.attention_masks[idx].squeeze(), self.label_ids[idx].squeeze()


# %%
if tokenizer.pad_token is None:
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})# 
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.config.dropout = 0.1
model.config.attention_dropout = 0.1

# Prepare datasets
train_dataset = MemoryAccessDataset(tokenizer, train_data)
validation_dataset = MemoryAccessDataset(tokenizer, validation_data)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=2)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 1

model.train()
for epoch in range(epochs):  # Define epochs
    print("epoch ", epoch)
    for batch in train_loader:
        
        inputs, masks, labels = batch
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# %%
def predict_next_page_address(model, tokenizer, prompt):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    
    # Create the prompt from the input 
    
    # Encode the prompt to be suitable for the model
    inputs = tokenizer.encode(prompt, return_tensors="pt", return_attention_mask=True).to(device)
    # print(inputs)
    # print("prompt :" + prompt)

    # Generate the output using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length = 16,
            min_length = 16,
            num_return_sequences=1,
            # pad_token_id=tokenizer.pad_token_id   
        )

    
    # Decode the generated output to text
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("Generated text: " + predicted_text)
    # Extract the next page address from the predicted text
    predicted_page_address = predicted_text.split()[-1] 
    truncated_predicted_text = predicted_page_address[:8]

    final_address = hex(int(truncated_predicted_text, 16) + 0x1000)
    clean_hex = final_address.replace('0x', '')

    return clean_hex

# # Example usage
# page_address = "-77547"  # Example current page address
# predicted_page_address = predict_next_page_address(model, tokenizer, page_address)
# print(f"Predicted Next Page Address: {predicted_page_address}")


# %%
def test_accuracy(model, tokenizer, data):
    correct_predictions = 0
    total_predictions = 0
    for index in range(len(data) - 1):
        prompt = f"{data[index]}"
        next_page_address = data[index + 1]

        # Generate prediction and immediately handle output
        predicted_page_address = predict_next_page_address(model, tokenizer, prompt)
        
        # Check if the prediction matches the actual address
        if predicted_page_address == next_page_address:
            correct_predictions += 1
        total_predictions += 1

        # Print results immediately after prediction to maintain order
        # print(f"Predicted: {predicted_page_address}  Actual: {next_page_address}")

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

accuracy = test_accuracy(model, tokenizer, validation_data)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
# Save the model
model_path = "/home/junting/data/gpt2-self"
model.save_pretrained(model_path)

# Save the tokenizer
tokenizer_path = "/home/junting/data/gpt2-self"
tokenizer.save_pretrained(tokenizer_path)


# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# # Example usage
# page_address = "-77547"  # Example current page address
# predicted_page_address = predict_next_page_address(model, tokenizer, page_address)
# print(f"Predicted Next Page Address: {predicted_page_address}")


