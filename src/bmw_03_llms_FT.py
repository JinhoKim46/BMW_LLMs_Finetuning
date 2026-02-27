import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import retrieve_config, save_file_dump

from .bmx_opp_base import Base

FT_MODEL_NAMES = {
    "none":"FT_full",
    "last":"FT-M1B",
}


class FineTuner(Base):
    def __init__(self, config_path, log_name="FineTuner", log_lv=0):
        super().__init__(config_path, log_name, log_lv)
        self.config_data = retrieve_config(self.config_path, "data")
        self.config_llms = retrieve_config(self.config_path, "llms")
        self.config_train = retrieve_config(self.config_path, "train")

        self.run_name = retrieve_config(self.config_path, "name")
        self.note = retrieve_config(self.config_path, "note")

    def prep_result_dir(self, log_lv=0) -> Path:
        if self.run_name is not None:
            self.logger.info(f"Run name for this training/evaluation: {self.run_name}", level=log_lv)
            result_dir_stem = self.run_name
        else:
            result_dir_stem = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.note is not None:
                self.logger.info(f"Note for this training/evaluation: {self.note}", level=log_lv)
                result_dir_stem += f"_{self.note}"

        out_dir = Path(f"{self.config_data.get('output_dir', 'results')}/{result_dir_stem}")
        if not out_dir.exists():  # Create out_dir folder
            out_dir.mkdir(parents=True, exist_ok=True)

        return out_dir

    def get_models(self, log_lv=0) -> Dict[str, AutoModelForCausalLM]:
        MODEL_NAME = self.config_llms.get("model", "gpt2")

        discard_layer_list = self.config_llms.get("variations", ["none"])

        models = {}
        for discard_layer in discard_layer_list:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
            )

            if discard_layer == "none":
                self.logger.info("Fine-tuning model with full depth was added!", level=log_lv)
                models[FT_MODEL_NAMES[discard_layer]] = model
                continue
            else:
                if discard_layer == "last":
                    discard_idx = -1
                else:
                    raise NotImplementedError(
                        "TODO: discard_layer for other options [first, mid, random] are not available now."
                    )

                del model.transformer.h[discard_idx]  # remove the last block
                model.config.n_layer = len(model.transformer.h)

                # TODO: extention to other options
                self.logger.info("Fine-tuning model with Minus 1 block (M1B) depth was added!", level=log_lv)
                models[FT_MODEL_NAMES[discard_layer]] = model

        return models

    def config_LoRA(self, model_name, model, log_lv=0):
        self.logger.info(f"[{model_name}] Parameters for the LoRA method", level=log_lv)
        print(json.dumps(self.config_llms.get("LoRA", {}), indent=4))

        # 2) Add LoRA (target GPT-2 attention projections)
        lora_cfg = LoraConfig(**self.config_llms.get("LoRA", {}))
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        return model

    def define_dataset(self, log_lv=0):
        data_path = f"{self.config_data.get('db_root', 'database')}/{self.config_data.get('prepare',{}).get('processed_data_fname', 'articles_proc.jsonl')}"

        ds = load_dataset("json", data_files=data_path)

        # TODO: for better flexibility, define test_size in config.yaml.
        ds = ds["train"].train_test_split(test_size=0.1, seed=self.seed)  # split the dataset into train and test sets
        train_ds, eval_ds = ds["train"], ds["test"]

        self.logger.info(f"Dataset loaded. Train size: {len(train_ds)}, Eval size: {len(eval_ds)}", level=log_lv)

        return train_ds, eval_ds

    def define_tokenizer(self, train_ds, eval_ds):
        def tok(batch):
            return tokenizer(
                batch["x"],
                **self.config_llms.get("tokenizer", {}),
            )

        tokenizer = AutoTokenizer.from_pretrained(self.config_llms.get("model", "gpt2-large"), use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        train_ds = train_ds.map(tok, batched=True, remove_columns=["x"])
        eval_ds = eval_ds.map(tok, batched=True, remove_columns=["x"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return train_ds, eval_ds, data_collator, tokenizer

    def get_trainer(self, model, train_ds, eval_ds, data_collator, out_dir: Path):
        train_args_cfg = dict(self.config_train)
        train_args_cfg["logging_dir"] = str(out_dir / "tb_logs")

        args = TrainingArguments(output_dir=str(out_dir), **train_args_cfg)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        return trainer

    def run(self, **kwargs):
        # TODO: if checkpoint is available in the out_dir, load the checkpoint and resume training. This is needed when the training needs to be stopped and restarted, as the training may take a long time.
        self.logger.info("Starting domain-adaptation of LLMs on the preprocessed article dataset...", level=self.log_lv)

        out_dir = self.prep_result_dir(log_lv=self.log_lv + 1)
        save_file_dump(out_dir)

        train_ds, eval_ds = self.define_dataset(log_lv=self.log_lv + 1)

        train_ds, eval_ds, data_collator, tokenizer = self.define_tokenizer(train_ds, eval_ds)

        models = self.get_models(log_lv=self.log_lv + 1)

        for model_name, model in models.items():
            out_dir_sub = out_dir / model_name
            if not out_dir_sub.exists():  # Create out_dir_sub folder
                out_dir_sub.mkdir(parents=True, exist_ok=True)

            model = self.config_LoRA(model_name, model, log_lv=self.log_lv + 2)
            trainer = self.get_trainer(model, train_ds, eval_ds, data_collator, out_dir_sub)

            trainer.train()

            trainer.evaluate()

            trainer.save_model(out_dir_sub)
            tokenizer.save_pretrained(out_dir_sub)

        return {"out_dir": out_dir, "tokenizer": tokenizer}


if __name__ == "__main__":
    fine_tuner = FineTuner(config_path="./config")
    fine_tuner.run()
