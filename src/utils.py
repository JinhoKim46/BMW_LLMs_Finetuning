import json
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

import pandas as pd
import yaml
from transformers import AutoTokenizer


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


def gen_md_table(result: str, summary: pd.DataFrame):
    bert_p_MinMax = (float(summary["bert_p"].min()), float(summary["bert_p"].max()))
    bert_r_MinMax = (float(summary["bert_r"].min()), float(summary["bert_r"].max()))
    bert_f1_MinMax = (float(summary["bert_f1"].min()), float(summary["bert_f1"].max()))
    nll_MinMax = (float(summary["nll"].min()), float(summary["nll"].max()))

    def _fmt(value: float, minmax: tuple, larger_is_better=True, epsilon=1e-12) -> str:
        min_val, max_val = minmax
        if larger_is_better:
            is_best = abs(value - max_val) < epsilon
        else:
            is_best = abs(value - min_val) < epsilon

        text = f"{value:.4f}"
        return f"**{text}**" if is_best else text

    result += "## Q&A BERTScore Summary\n\n"
    result += "| model | nll | bert_p | bert_r | bert_f1 |\n"
    result += "|---|---:|---:|---:|---:|\n"
    for _, row in summary.iterrows():
        result += (
            f"| {row['model']} | "
            f"{_fmt(float(row['nll']), nll_MinMax, False)} | "
            f"{_fmt(float(row['bert_p']), bert_p_MinMax)} | "
            f"{_fmt(float(row['bert_r']), bert_r_MinMax)} | "
            f"{_fmt(float(row['bert_f1']), bert_f1_MinMax)} |\n"
        )
    result += "\n"

    return result
    return result


def prep_eval_wo_ft(config_path):
    config_data = retrieve_config(config_path, "data")
    config_llms = retrieve_config(config_path, "llms")
    run_name = retrieve_config(config_path, "name")
    out_dir = Path(config_data["output_dir"]) / str(run_name)

    tokenizer = AutoTokenizer.from_pretrained(config_llms.get("model", "gpt2"), use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    return {"out_dir": out_dir, "tokenizer": tokenizer}
