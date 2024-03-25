from pathlib import Path
from typing import Tuple
import random
import json

from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
import numpy as np

from args import Args
from model.meem import Meem


def get_output_dir(args: Args) -> Path:
    training_name = f"ep{args.num_epochs}-ee{args.early_exit}-lr{args.lr}"
    data_name = f"para{args.use_paraphrases}_neg{args.use_neg_examples}"
    if args.train_params == "entity-mem":
        prompt_args_name = f"rp{args.reparam_prompt}-pl{int(args.prompt_len)}"
        output_dir = Path(
            "result",
            args.pretrained_name,
            args.train_params,
            prompt_args_name,
            data_name,
            training_name,
        )
    else:
        raise ValueError
        # output_dir = Path(
        #     "result", args.pretrained_name, args.train_params, training_name
        # )
    return output_dir


def init_meem(args: Args) -> Tuple[Meem, T5TokenizerFast]:
    print("Loading tokenizer and model from:", args.pretrained_name)
    tokenizer = T5TokenizerFast.from_pretrained(
        args.pretrained_name, cache_dir=args.cache_dir
    )
    model = Meem(
        num_entities=args.num_examples,
        pretrained_name=args.pretrained_name,
        prompt_len=args.prompt_len,
        # d_ent=512,
        # d_reparam_hidden=512,
        reparam_prompt=bool(args.reparam_prompt),
        cache_dir=args.cache_dir,
    )
    return model, tokenizer


def load_ckpt(model: torch.nn.Module, path: Path):
    print(f"Loading checkpoint from {path}")
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_t5(
    pretrained_name: str, cache_dir: str
) -> Tuple[T5ForConditionalGeneration, T5TokenizerFast]:
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_name, cache_dir=cache_dir
    )
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_name, cache_dir=cache_dir)
    if isinstance(model, T5ForConditionalGeneration):
        return model, tokenizer
    else:
        raise ValueError


class Logfile:
    def __init__(self, path: Path):
        self.path = path

    def clear(self):
        open(self.path, "w", encoding='utf8').write("")

    def log(self, text: str):
        print(text)
        with open(self.path, "a", encoding='utf8') as out:
            out.write(text + "\n")


def load_json(path):
    return json.load(open(path, "r", encoding="utf8"))


def get_data_name_and_task_type(args: Args) -> Tuple[str, str]:
    data_path = Path(args.data_path)
    data_name = data_path.stem

    # Task type
    if "cf" in data_name:
        task_type = "mrc"
    elif "fever" in data_name:
        if bool(args.nli_with_mrc):
            task_type = "mrc"
        else:
            task_type = "nli"
    elif "hotpot" in data_name:
        task_type = "mrc"
    else:
        raise ValueError(f"Invalid data name: {data_name}")

    if bool(args.mrc_convert_to_bool_q):
        assert "cf" in data_name
        data_name += "_bool"
    if bool(args.nli_with_mrc):
        assert "fever" in data_name
        data_name += "_mrc"
    return data_name, task_type
