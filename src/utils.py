import json
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

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


def save_file_dump(output_dir: Path):
    save_path = output_dir / "script_dump"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    dirs = ["config", "src"]
    for dir in dirs:
        try:
            copytree(dir, save_path / dir, gnore=ignore_patterns("__pycache__"))
        except:
            pass
    files = ["main.py"]
    for file in files:
        try:
            copy2(file, save_path / file)
        except:
            print(f"{file} does not exist.")
