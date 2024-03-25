import json
from typing import List, Dict, Optional

from transformers import T5TokenizerFast, BatchEncoding
from torch import Tensor
import torch


def load_lines(path):
    return [line.strip() for line in open(path, "r", encoding="utf8")]


def dump_lines(path, data):
    with open(path, "w", encoding="utf8") as f:
        for item in data:
            f.write(item + "\n")


def load_json(path):
    return json.load(open(path, "r", encoding="utf8"))


def iter_jsonl(path, cnt: Optional[int] = None):
    i = 0
    for line in open(path, "r", encoding="utf8"):
        yield json.loads(line)
        i += 1
        if cnt is not None and i >= cnt:
            break


def load_jsonl(path, cnt: Optional[int] = None):
    return list(iter_jsonl(path, cnt))


def dump_json(data, path):
    json.dump(data, open(path, "w", encoding="utf8"), indent=2, ensure_ascii=False)


def dump_jsonl(data, path: str):
    with open(path, "w", encoding="utf8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_flan_input(input_text: str) -> str:
    input_text = input_text.strip()
    if input_text.endswith("?"):
        input_text = input_text[:-1]
    input_text = "Q: {}? A:".format(input_text)
    return input_text


def mask_triple(
    triple: List[str],
    ent_to_id: Dict[str, int],
    mask_token="<extra_id_0>",
) -> List[list]:
    """
    Mask the triple by replacing the subject or object with the mask token.

    "China capital Beijing"
    ->
    [
        ["China capital <extra_id_0>", "Beijing"],
        ["<extra_id_0> capital Beijing", "China"],
    ]

    Also, if, for instance, "Beijing" is not in `ent_to_id`, then the
    example where it is the label (and "China" is masked) is discarded.

    NOTE: <extra_id_0> is the mask token used for text-infilling in T5.
    See: https://github.com/huggingface/transformers/issues/3985
    """
    subj, rel, obj = triple
    masked = [[ent_to_id[subj], f"{subj} {rel} {mask_token}", obj]]
    # if obj in ent_to_id:
    #     masked.append([ent_to_id[obj], f"{mask_token} {rel} {obj}", subj])
    return masked


def preprocess_input(text: str) -> str:
    """
    Remove garbage from the input text in the CounterFact dataset.
    """
    text = text.strip()
    while ":" in text:
        i = text.find(":")
        j = i + 1
        while j < len(text) and text[j] != " ":
            j += 1
        text = text[j:]
    while "\n" in text:
        text = text[text.find("\n") + 1 :].strip()
    while ". " in text:
        text = text[text.find(". ") + 2 :].strip()

    if text.endswith("?"):
        text = text[:-1].strip()

    redundant_after_question_mark = [
        " It is", " It was", " It's",
    ]
    for s in redundant_after_question_mark:
        if text.endswith(s):
            text = text[:-len(s)]

    if text.endswith("an"):
        text = text[:-1]

    if text.endswith("by"):
        text += " who"
    if text.endswith("in"):
        text += " where"
    for what_ending in ["is", "as", "a", "on", "at", "for", "of", "the"]:
        if text.endswith(what_ending):
            text += " what"
            break

    text += "?"
    return text


def gen_training_features(
    tokenizer: T5TokenizerFast,
    example: dict,
    flan_format: bool,
    use_paraphrases: bool,
    use_neg_examples: bool,
) -> BatchEncoding:
    """Return a list of features for training, one for each edit."""
    input_text = example["input"]
    output_text = example["output_new"]
    input_texts = [input_text]
    if use_paraphrases:
        input_texts += example["paraphrases_generated"]
    if use_neg_examples:
        print(len(example["unrelated_same_subject"]))
        exit()
    output_texts = [output_text] * len(input_texts)
    ent_id = example["id"]
    ent_ids = torch.ones(len(input_texts), dtype=torch.long) * ent_id
    if flan_format:
        input_texts = [format_flan_input(x) for x in input_texts]
    encoded = tokenizer(
        input_texts,
        text_target=output_texts,
        padding="longest",
        return_tensors="pt",
    )
    encoded["ent_ids"] = ent_ids
    return encoded


def gen_eval_features(
    tokenizer: T5TokenizerFast,
    input_texts: List[str],
    ent_id: Optional[int] = None,
) -> BatchEncoding:
    """
    Return a list of features, one for each edit.

    `ent_id`: if not None, the feature BatchEncoding will have an "ent_id" field.
    """
    encoded = tokenizer(
        input_texts,
        padding="longest",
        return_tensors="pt",
    )
    if ent_id is not None:
        ent_ids = torch.ones(len(input_texts), dtype=torch.long) * ent_id
        encoded["ent_ids"] = ent_ids
    return encoded


def pack_batches(features, batch_size: int = 32) -> List[Dict[str, Tensor]]:
    """Pack features into batches."""
    batched_features = []
    num_features = features["input_ids"].size(0)
    for i in range(0, num_features, batch_size):
        batched_features.append({k: v[i : i + batch_size] for k, v in features.items()})
    return batched_features
