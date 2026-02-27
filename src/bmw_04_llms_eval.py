import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from bert_score import score as bert_score
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import gen_md_table, retrieve_config
from .bmx_opp_base import Base


class Evaluation(Base):
    def __init__(self, config_path, log_name="Evaluation", log_lv=0):
        super().__init__(config_path, log_name, log_lv)
        self.config_eval = retrieve_config(self.config_path, "eval")
        self.config_llms = retrieve_config(self.config_path, "llms")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # @@@@@@@@@@@@@@@@@@@@@@@@
    # BMW-related text generation
    def get_comparing_models(
        self, tokenizer: AutoTokenizer, out_dir: Path, base_model_name: str
    ) -> List[Tuple[str, AutoModelForCausalLM]]:
        # Assume that result directories are in out_dir/model_name
        ft_model_roots = [p for p in out_dir.iterdir() if p.is_dir() if p.stem not in ["script_dump"]]
        comparing_models = []

        base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto").to(self.device)
        base.config.pad_token_id = tokenizer.eos_token_id
        base.eval()
        comparing_models.append(("base", base))

        for ft_model_root in ft_model_roots:
            ft_model = AutoPeftModelForCausalLM.from_pretrained(ft_model_root, torch_dtype="auto").to(self.device)
            ft_model.config.pad_token_id = tokenizer.eos_token_id
            ft_model.eval()
            comparing_models.append((ft_model_root.stem, ft_model))

        return comparing_models

    def generate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inputs, **self.config_eval.get("generation", {}))
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def eval_text_gen(
        self,
        prompts: List[str],
        models: List[Tuple[str, AutoModelForCausalLM]],
        tokenizer: AutoTokenizer,
        out_dir: Path,
    ):

        result = "# BMW-related text Generation\n\n"

        for i, p in enumerate(prompts, start=1):
            result += f"## Prompt: {i}: {p}\n\n"
            for model_name, model in models:
                gen_text = self.generate(model, tokenizer, p)
                result += f"### Model: {model_name}\n\n"
                result += f"{gen_text}\n\n"

        with open(out_dir / "report_inference.md", "a", encoding="utf-8") as f:
            f.write(result)

    # @@@@@@@@@@@@@@@@@@@@@@@
    # BMW-related Q&A
    def load_qna(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def extract_answer(self, full_text):
        if "A:" in full_text:
            return full_text.split("A:", 1)[1].strip().split("\n")[0]
        return full_text.strip().split("\n")[0]

    def generate_answer(self, model: AutoModelForCausalLM, question: str, tokenizer: AutoTokenizer):
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        eval_gen_cfg = dict(self.config_eval.get("qna", {}))  # Use separate generation config for QnA if available
        eval_gen_cfg["pad_token_id"] = tokenizer.eos_token_id
        eval_gen_cfg["eos_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = model.generate(**inputs, **eval_gen_cfg)

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self.extract_answer(decoded)

    def compute_nll(self, model: AutoModelForCausalLM, question: str, answer: str, tokenizer: AutoTokenizer) -> float:
        """Compute the negative log-likelihood of the reference answer given the question."""
        prompt = f"Q: {question}\nA: {answer}"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Find where the answer tokens start
        prompt_only = f"Q: {question}\nA:"
        prompt_ids = tokenizer(prompt_only, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            # outputs.loss is the mean cross-entropy over ALL tokens; recompute over answer tokens only
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, prompt_len - 1 : -1, :].contiguous()
        shift_labels = input_ids[:, prompt_len:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        nll = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return float(nll.item())

    def eval_QnA(
        self, QnA: List[str], models: List[Tuple[str, AutoModelForCausalLM]], tokenizer: AutoTokenizer, out_dir: Path
    ):

        result = "# BMW-related Q&A\n\n"

        all_results = []
        for model_name, model in models:
            result_per_model = []

            # TODO: Batch processing to compute metrics, not per question-answer pair.
            for qna in QnA:
                pred = self.generate_answer(model, qna["question"], tokenizer)
                ref = qna["answer"]

                p, r, f1 = bert_score(
                    [pred],
                    [ref],
                    lang="en",
                    model_type="distilbert-base-uncased",
                    device=self.device,
                    verbose=False,
                )

                nll = self.compute_nll(model, qna["question"], ref, tokenizer)

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
                        "nll": nll,
                    }
                )

            all_results.extend(result_per_model)
        all_results_df = pd.DataFrame(all_results)
        model_order = [name for name, _ in models]
        summary = (
            all_results_df.assign(model=pd.Categorical(all_results_df["model"], categories=model_order, ordered=True))
            .groupby("model", sort=False, observed=True)
            .agg(
                nll=("nll", "mean"),
                bert_p=("bert_p", "mean"),
                bert_r=("bert_r", "mean"),
                bert_f1=("bert_f1", "mean"),
            )
            .reset_index()
        )
        result = gen_md_table(result, summary)
        with open(out_dir / "report_inference.md", "a", encoding="utf-8") as f:
            f.write(result)

    def run(self, out_dir: Path, tokenizer: AutoTokenizer, **kwargs):
        self.logger.info("Starting evaluation of LLMs on BMW-related tasks...", level=self.log_lv)

        base_model_name = self.config_llms.get("model", "gpt2")
        comparing_models = self.get_comparing_models(tokenizer, out_dir, base_model_name)

        prompt_root = self.config_eval.get("prompt_dir", "./prompts")
        with open(f"{prompt_root}/prompt.txt", "r", encoding="utf-8") as f:
            prompts = [i.strip() for i in f.readlines()]
        self.logger.info(f"Loaded {len(prompts)} prompts for evaluation.", level=self.log_lv)
        self.eval_text_gen(prompts, comparing_models, tokenizer, out_dir)

        qna_path = Path(prompt_root) / "qna.jsonl"
        if not qna_path.exists():
            qna_path = Path(prompt_root) / "QnA.jsonl"

        qna = self.load_qna(qna_path)
        self.logger.info(f"Loaded {len(qna)} Q&A pairs for evaluation.", level=self.log_lv)
        self.eval_QnA(qna, comparing_models, tokenizer, out_dir)


if __name__ == "__main__":   
    out_dir = Path("/mnt/jinho/Development/Projects/2026/BMW_LLMs_finetuning/results/20260225_091140")

    config_llms = retrieve_config("./config/config.yaml", "llms")
    tokenizer = AutoTokenizer.from_pretrained(config_llms.get("model", "gpt2"), use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    eval = Evaluation(config_path="./config", log_lv=1)
    eval.run(out_dir, tokenizer)
