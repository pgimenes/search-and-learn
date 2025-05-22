import json

# Load the JSONL file
file_path = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_completions-hp.jsonl"  # Replace with your actual file path

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))  # Load each line as a dictionary

# Print the first entry to verify
breakpoint()