import json

import yaml


def retrieve_config(config_path, key):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get(key, None)

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                records.append(json.loads(line))
    return records