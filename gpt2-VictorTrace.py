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
with open('./data/raw/fltrace_out/canneal/300_180/fltrace-data-faults-19714-1.out', 'r') as f:
    count = 0
    for line in f:
        parts = line.strip().split(',')
        trace = parts[-1]
        addresses = trace.split('|')
        data.append(addresses)
        count += 1
        if count == 5:
            break
            
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
validation_data = data[split_index:]

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
        self.sequence_length = sequence_length
        temp_in_sequence = ''
        temp_label_sequence = ''
        for trace in data:
            if trace == 'trace':
                continue
            for i in range (len(trace) - 2):
                # temp_in_sequence += f"current_address: {trace[i]}  next_address: {trace[i + 1]}"
                temp_in_sequence += f"{trace[i]},"
                temp_label_sequence += f"{trace[i + 1]},"

                if (i + 1) % sequence_length == 0:
                    encoding_in = tokenizer(temp_in_sequence, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
                    encoding_label = tokenizer(temp_label_sequence, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
                    self.input_ids.append(encoding_in['input_ids'])
                    self.label_ids.append(encoding_label['input_ids'])
                    self.attention_masks.append(encoding_in['attention_mask'])
                    print(f"in_sequence: {temp_in_sequence}")
                    print(f"lable_sequence: {temp_label_sequence}")
                    temp_in_sequence = ""
                    temp_label_sequence = ""

        if temp_in_sequence:
                encoding_in = tokenizer(temp_in_sequence, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
                encoding_label = tokenizer(temp_label_sequence, max_length=512, padding='max_length', truncation=True, return_tensors="pt", return_attention_mask=True)
                self.input_ids.append(encoding_in['input_ids'])
                self.label_ids.append(encoding_label['input_ids'])
                self.attention_masks.append(encoding_in['attention_mask'])
                temp_in_sequence = ""
                temp_label_sequence = ""

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # print(f" encode check : {self.label_ids[idx].squeeze() == self.input_ids[idx].squeeze()}")
        print(f"input {self.input_ids[idx].squeeze()}")
        print(f"label {self.label_ids[idx].squeeze()}")
        return self.input_ids[idx].squeeze(), self.attention_masks[idx].squeeze(), self.label_ids[idx].squeeze()

# %%
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        # print(f"inputs : {inputs}")
        # print(f"labels : {labels}")
        # print(inputs == labels)

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()



# %%
def predict_next_page_address(model, tokenizer, page_address):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    
    # Create the prompt from the input PC and page address
    prompt = f" Address: {page_address}"
    
    # Encode the prompt to be suitable for the model
    inputs = tokenizer.encode(prompt, return_tensors="pt", return_attention_mask=True).to(device)

    # Generate the output using the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    
    # Decode the generated output to text
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the next page address from the predicted text
    predicted_page_address = predicted_text.split()[-1] 

    return predicted_page_address

# # Example usage
# page_address = "-77547"  # Example current page address
# predicted_page_address = predict_next_page_address(model, tokenizer, page_address)
# print(f"Predicted Next Page Address: {predicted_page_address}")


# %%
# test the accuracy

def test_accuracy(model, tokenizer, data):
    correct_predictions = 0
    total_predictions = 0
    for trace in data:
        # print(trace)
        # print(trace[0])
        for i in range(len(trace) - 2):
            page_address = trace[i]
            # print(page_address)
            # print('type of page_address' + str(type(page_address)))

            next_page_address = trace[i+1]
            # print(next_page_address)
            # print('type of next_page_address' + str(type(next_page_address)))

            predicted_page_address = predict_next_page_address(model, tokenizer, page_address)
            # print(predicted_page_address)
            # print('type of predicted_page_address ' + str(type(predicted_page_address)))

            if predicted_page_address == next_page_address:
                correct_predictions += 1
            total_predictions += 1
            # print(total_predictions)
            # print('type of total_predictions ' + str(type(total_predictions)))
            print(f"Predicted: {predicted_page_address}  Actual: {next_page_address}")
    accuracy = correct_predictions / total_predictions
    return accuracy

accuracy = test_accuracy(model, tokenizer, validation_data)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
# Save the model
model_path = "/home/junting/data/gpt2_trained_model"
model.save_pretrained(model_path)

# Save the tokenizer
tokenizer_path = "/home/junting/data/gpt2_trained_tokenizer"
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


