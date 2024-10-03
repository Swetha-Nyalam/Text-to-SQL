#!/bin/bash

db_path="uber_rides.db"
golden_file="Data/gold_data_set.jsonl"
db_type="SQLite"
db_name="uber"
prompt_types="leasttomost"

python3 pipeline.py --db_path "$db_path" --golden_file "$golden_file" --db_type "$db_type" --db_name "$db_name" --prompt_types "$prompt_types"