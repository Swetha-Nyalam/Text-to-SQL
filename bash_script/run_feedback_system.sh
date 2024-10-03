#!/bin/bash

model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
db_path="uber_rides.db"
input_file="Data/train_data.jsonl"
output_file="Data/train_data.jsonl"

# run the python script
python enhanced_train_data.py "$model_name" "$db_path" "$input_file" "$output_file"

