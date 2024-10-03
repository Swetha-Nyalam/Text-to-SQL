#!/bin/bash

# set the required arguments
lamini_api_key="a6b5f98f73759c6bd5429214bb9eee4a20c4a56f11e411554021206742a7efdd"
training_data_path="Data/train_data_feedback.jsonl"
model_name="meta-llama/meta-llama-3-8b-instruct"

# define the learning rate and max steps as defaults, can be modified if needed
learning_rate=0.0003
max_steps=600

# run the python script
python3 train_jobs.py \
    --lamini_api_key "$lamini_api_key" \
    --training_data_path "$training_data_path" \
    --model_name "$model_name" \
    --learning_rate "$learning_rate" \
    --max_steps "$max_steps"