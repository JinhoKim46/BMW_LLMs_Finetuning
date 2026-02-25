import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

from logger import get_logger
from utils import retrieve_config, save_file_dump

LOGGER = get_logger("llms - fine-tuning")

CONFIG_PATH = Path(__file__).parent.parent / "config/config.yaml"
CONFIG_DATA = retrieve_config(CONFIG_PATH, "data")
CONFIG_LLMS = retrieve_config(CONFIG_PATH, "llms")
CONFIG_TRAIN = retrieve_config(CONFIG_PATH, "train")

SEED = retrieve_config(CONFIG_PATH, "seed")
set_seed(SEED)  # Set random seed for reproducibility

FT_MODEL_NAMES = {
    "none":"FT_full",
    "last":"FT-M1B",
}


def prep_result_dir():
    run_name = retrieve_config(CONFIG_PATH, "name")
    if run_name is not None:
        LOGGER.info(f"Run name for this training/evaluation: {run_name}", level=1)
        result_dir_stem = run_name
    else:
        result_dir_stem = datetime.now().strftime("%Y%m%d_%H%M%S")

        note = retrieve_config(CONFIG_PATH, "note")
        if note is not None:
            LOGGER.info(f"Note for this training/evaluation: {note}", level=1)
            result_dir_stem += f"_{note}"

    out_dir = Path(f"{CONFIG_DATA.get('output_dir', 'results')}/{result_dir_stem}")
    if not out_dir.exists():  # Create out_dir folder
        out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def get_models() -> Dict[str, AutoModelForCausalLM]:
    MODEL_NAME = CONFIG_LLMS.get("model", "gpt2")
    
    discard_layer_list = CONFIG_LLMS.get("variations", ["none"])
    
    models = {}
    for discard_layer in discard_layer_list:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
        )
            
        if discard_layer == "none":
            LOGGER.info("Fine-tuning model with full depth was added!", level=1)
            models[FT_MODEL_NAMES[discard_layer]] = model
            continue
        else:
            if discard_layer == "last":
                discard_idx = -1
            else:
                raise NotImplementedError("TODO: discard_layer for other options [first, mid, random] are not available now.")
            
            del model.transformer.h[discard_idx]  # remove the last block
            model.config.n_layer = len(model.transformer.h)
            
            # TODO: extention to other options
            LOGGER.info("Fine-tuning model with Minus 1 block (M1B) depth was added!", level=1)
            models[FT_MODEL_NAMES[discard_layer]] = model 
    
    return models

def config_LoRA(model_name, model):
    LOGGER.info(f"[{model_name}] Parameters for the LoRA method", level=1)
    print(json.dumps(CONFIG_LLMS.get("LoRA", {}), indent=4))

    # 2) Add LoRA (target GPT-2 attention projections)
    lora_cfg = LoraConfig(**CONFIG_LLMS.get("LoRA", {}))
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model


def define_dataset():
    data_path = f"{CONFIG_DATA.get('db_root', 'database')}/{CONFIG_DATA.get('prepare',{}).get('processed_data_fname', 'articles_proc.jsonl')}"

    ds = load_dataset("json", data_files=data_path)

    # TODO: for better flexibility, define test_size in config.yaml.
    ds = ds["train"].train_test_split(test_size=0.1, seed=SEED)  # split the dataset into train and test sets
    train_ds, eval_ds = ds["train"], ds["test"]

    return train_ds, eval_ds


def define_tokenizer(train_ds, eval_ds):
    def tok(batch):
        return tokenizer(
            batch["x"],
            **CONFIG_LLMS.get("tokenizer", {}),
        )


    tokenizer = AutoTokenizer.from_pretrained(CONFIG_LLMS.get("model", "gpt2-large"), use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(tok, batched=True, remove_columns=["x"])
    eval_ds = eval_ds.map(tok, batched=True, remove_columns=["x"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    return train_ds, eval_ds, data_collator, tokenizer

def get_trainer(model, train_ds, eval_ds, data_collator, out_dir:Path):
    train_args_cfg = dict(CONFIG_TRAIN)
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


def fine_tuning():
    # TODO: if checkpoint is available in the out_dir, load the checkpoint and resume training. This is needed when the training needs to be stopped and restarted, as the training may take a long time.
    out_dir = prep_result_dir()
    save_file_dump(out_dir)

    train_ds, eval_ds = define_dataset()

    train_ds, eval_ds, data_collator, tokenizer = define_tokenizer(train_ds, eval_ds)

    models = get_models()

    for model_name, model in models.items():
        out_dir_sub = out_dir / model_name
        if not out_dir_sub.exists():  # Create out_dir_sub folder
            out_dir_sub.mkdir(parents=True, exist_ok=True)

        model = config_LoRA(model_name, model)
        trainer = get_trainer(model, train_ds, eval_ds, data_collator, out_dir_sub)

        trainer.train()

        trainer.evaluate()

        trainer.save_model(out_dir_sub)
        tokenizer.save_pretrained(out_dir_sub)

    return {"out_dir": out_dir, "tokenizer": tokenizer}

if __name__ == "__main__":
    _ = fine_tuning()
