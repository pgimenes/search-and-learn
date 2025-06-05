import json
import re

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

# Example usage
file_path = 'data/meta-llama/Llama-3.2-1B-Instruct/beam_search_completions.jsonl'
entries = load_jsonl(file_path)

# Example: print the first entry
correct = 0
incorrect = 0
not_found = 0

for idx, item in enumerate(entries):
    print(f"{idx}/{len(entries)}")

    pred = item["pred"]
    answer = item["answer"].replace(" ", "")
    match = re.search(r"\$\\boxed\s*{(.*?)}\$", pred)

    if match:
        candidate = match.group(1).replace(" ", "")

        if candidate == answer:
            print(f"correct")
            correct += 1
        else:
            print(f"incorrect. candidate: {candidate}, answer: {answer}")
            incorrect += 1
    else:
        print(f"nothing found")
        not_found += 1

acc = correct / len(entries)
print(f"ACCURACY: {acc}")
print(f"not found: {not_found / len(entries)}")