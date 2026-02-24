import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from bert_score import score as bert_score
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from logger import get_logger
from utils import retrieve_config

LOGGER = get_logger("llms - evaluation")

CONFIG_PATH = Path(__file__).parent.parent / "config/config.yaml"
CONFIG_GEN = retrieve_config(CONFIG_PATH, "generation")
CONFIG_LLMS = retrieve_config(CONFIG_PATH, "llms")
CONFIG_DATA = retrieve_config(CONFIG_PATH, "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
set_seed(retrieve_config(CONFIG_PATH, "seed"))  # Set random seed for reproducibility


# @@@@@@@@@@@@@@@@@@@@@@@@
# BMW-related text generation
def get_comparing_models(tokenizer:AutoTokenizer, out_dir:Path, base_model_name: str) -> List[Tuple[str, AutoModelForCausalLM]]:
    # Assume that result directories are in out_dir/model_name
    ft_model_roots = [p for p in out_dir.iterdir() if p.is_dir()]
    comparing_models = []   
    
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto").to(DEVICE)
    base.config.pad_token_id = tokenizer.eos_token_id
    base.eval()
    comparing_models.append(("base", base))

    for ft_model_root in ft_model_roots:
        ft_model = AutoPeftModelForCausalLM.from_pretrained(ft_model_root, torch_dtype="auto").to(DEVICE)
        ft_model.config.pad_token_id = tokenizer.eos_token_id
        ft_model.eval()
        comparing_models.append((ft_model_root.stem, ft_model))
    
    return comparing_models

def generate(model: AutoModelForCausalLM, tokenizer:AutoTokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, **CONFIG_GEN)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def eval_text_gen(prompts: List[str], models:List[Tuple[str, AutoModelForCausalLM]], tokenizer:AutoTokenizer, out_dir:Path):

    result = "# BMW-related text Generation\n\n"
    
    for i, p in enumerate(prompts, start=1):
        result += f"## Prompt: {i}: {p}\n\n"
        for model_name, model in models:
            gen_text = generate(model, tokenizer, p)
            result += f"### Model: {model_name}\n\n"
            result += f"{gen_text}\n\n"
            
    with open(out_dir / "report_inference.md", "a") as f:
        f.write(result)



# @@@@@@@@@@@@@@@@@@@@@@@
# BMW-related Q&A
def load_qna(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_answer(full_text):
    if "A:" in full_text:
        return full_text.split("A:", 1)[1].strip().split("\n")[0]
    return full_text.strip().split("\n")[0]


def generate_answer(model, question):
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    eval_gen_cfg = dict(CONFIG_GEN)
    eval_gen_cfg["do_sample"] = False
    eval_gen_cfg["max_new_tokens"] = min(eval_gen_cfg.get("max_new_tokens", 64), 32)
    eval_gen_cfg["pad_token_id"] = tokenizer.eos_token_id
    eval_gen_cfg["eos_token_id"] = tokenizer.eos_token_id
    eval_gen_cfg.pop("temperature", None)
    eval_gen_cfg.pop("top_p", None)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **eval_gen_cfg)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_answer(decoded)

def compute_bertscore(df, model_name, scorer_model="distilbert-base-uncased"):
    cands = df["prediction"].tolist()
    refs = df["reference"].tolist()

    p, r, f1 = bert_score(
        cands,
        refs,
        lang="en",
        model_type=scorer_model,
        device=DEVICE,
        verbose=False,
    )

    return {
        "model": model_name,
        "bert_p": float(p.mean().item()),
        "bert_r": float(r.mean().item()),
        "bert_f1": float(f1.mean().item()),
    }


def eval_QnA(QnA: List[str], models:List[Tuple[str, AutoModelForCausalLM]], tokenizer:AutoTokenizer, out_dir:Path):

    result = "# BMW-related Q&A\n\n"
    
    all_results = []    
    for model_name, model in models:
        result_per_model = []
        for qna in QnA:
            pred = generate_answer(model, qna["question"])
            ref = qna["answer"]

            p, r, f1 = bert_score(
                [pred],
                [ref],
                lang="en",
                model_type="distilbert-base-uncased",
                device=DEVICE,
                verbose=False,
            )

            result_per_model.append(
                {
                    "model": model_name,
                    "id": qna["id"],
                    "question": qna["question"],
                    "reference": ref,
                    "prediction": pred,
                    "bert_p": float(p.mean().item()),
                    "bert_r": float(r.mean().item()),
                    "bert_f1": float(f1.mean().item()),
                }
            )

        all_results.extend(result_per_model)            
    all_results_df = pd.DataFrame(all_results)
    summary = (
        all_results_df.groupby("model", as_index=False)
        .agg(bert_p=("bert_p", "mean"), bert_r=("bert_r", "mean"), bert_f1=("bert_f1", "mean"))
        .sort_values("bert_p")
    )

    max_bert_p = summary["bert_p"].max()
    max_bert_r = summary["bert_r"].max()
    max_bert_f1 = summary["bert_f1"].max()

    def _fmt(value: float, max_value: float) -> str:
        text = f"{value:.4f}"
        return f"**{text}**" if abs(value - max_value) < 1e-12 else text

    result += "## Q&A BERTScore Summary\n\n"
    result += "| model | bert_p | bert_r | bert_f1 |\n"
    result += "|---|---:|---:|---:|\n"
    for _, row in summary.iterrows():
        result += (
            f"| {row['model']} | "
            f"{_fmt(float(row['bert_p']), float(max_bert_p))} | "
            f"{_fmt(float(row['bert_r']), float(max_bert_r))} | "
            f"{_fmt(float(row['bert_f1']), float(max_bert_f1))} |\n"
        )
    result += "\n"

    with open(out_dir / "report_inference.md", "a") as f:
        f.write(result)
        
        
def evaluation(out_dir: Path, tokenizer: AutoTokenizer):
    base_model_name = CONFIG_LLMS.get("model", "gpt2")
    comparing_models = get_comparing_models(tokenizer, out_dir, base_model_name)
    
    db_root = CONFIG_DATA.get("db_root", "database")
    with open(f"{db_root}/prompt.txt", "r") as f:
        prompts = [i.strip() for i in f.readlines()]    
    LOGGER.info(f"Loaded {len(prompts)} prompts for evaluation.")
    eval_text_gen(prompts, comparing_models, tokenizer, out_dir)
    
    qna_path = Path(db_root) / "qna.jsonl"
    if not qna_path.exists():
        qna_path = Path(db_root) / "QnA.jsonl"

    qna = load_qna(qna_path)
    LOGGER.info(f"Loaded {len(qna)} Q&A pairs for evaluation.")
    eval_QnA(qna, comparing_models, tokenizer, out_dir)
    
    
if __name__ == "__main__":   
    out_dir = Path("/mnt/jinho/Development/Projects/2026/BMW_LLMs_finetuning/results/20260224_134636")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_LLMS.get("model", "gpt2"), use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    evaluation(out_dir, tokenizer)