from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Sequence

from torch.utils.data import Dataset

if __name__ == "__main__":
    from utils import load_json, preprocess_input
else:
    from .utils import load_json, preprocess_input


def load_examples_same_subject(
    data_path: Path, num_examples: Optional[int] = None
) -> List[List[Tuple[int, str, str]]]:
    """
    Load examples from preprocessed file containing edits with the same subject.
    """
    edits = load_json(data_path)
    edits = edits[:num_examples]
    examples = []
    for edit in edits:
        entity_id = edit["case_id"]  # TODO: fix this
        prompts = edit["prompts"]
        for prompt in prompts:
            if prompt["rel_id"] == "P1412":
                continue
            cur_examples = []
            input_texts = prompt["inputs"]
            output_text = prompt["output"]
            for input_text in input_texts:
                input_text = preprocess_input(input_text)
                cur_examples.append((entity_id, input_text, output_text))
            examples.append(cur_examples)
    return examples


def load_examples(
    data_dir: Path, num_examples: Optional[int] = None
) -> List[Dict[str, Union[str, List[str], List[dict]]]]:
    def format_many_inputs(texts: List[str]) -> List[str]:
        texts = list(dict.fromkeys(texts))
        return [preprocess_input(text) for text in texts]

    data_path = data_dir / "counterfact.json"
    same_subjects_path = data_dir / "edits_same_subject.json"
    templates_path = data_dir / "templates.json"
    print("Loading data")
    examples_same_subject = load_json(same_subjects_path)
    templates = load_json(templates_path)
    templates_set = set()
    for prompts in templates.values():
        for prompt in prompts:
            templates_set.add(prompt)

    entries = load_json(data_path)
    entries = entries[:num_examples]
    examples = []
    print("Processing data")
    for idx, eg in enumerate(entries):
        rewrite = eg["requested_rewrite"]
        subject_name = rewrite["subject"]
        if rewrite["prompt"] not in templates_set:
            continue
        example = {
            "id": eg["case_id"],
            "input": preprocess_input(rewrite["prompt"].format(subject_name)),
            "output_true": rewrite["target_true"]["str"],
            "output_new": rewrite["target_new"]["str"],
            "subject_name": subject_name,
            "subject_id": None,
            "paraphrases": format_many_inputs(eg["paraphrase_prompts"]),
            "paraphrases_generated": format_many_inputs(eg["generation_prompts"]),
            "unrelated_diff_subject": format_many_inputs(eg["neighborhood_prompts"]),
            # "unrelated_diff_subject_outputs": [],
        }

        # Get unrelated facts with the same subject
        eg_same_subject = examples_same_subject[idx]
        all_prompts_same_subject: List[dict] = eg_same_subject["prompts"]
        unrelated_same_subject = []
        for prompt_data in all_prompts_same_subject:
            inputs: List[str] = prompt_data["inputs"]
            # Skip all unrelated examples whose output is the same as the new output
            if prompt_data["output"] == example["output_new"]:
                continue
            for i, prompt in enumerate(inputs):
                inputs[i] = preprocess_input(prompt)
            unrelated_same_subject.append(prompt_data)
        example["unrelated_same_subject"] = unrelated_same_subject
        examples.append(example)
    return examples


def process_edits(
    edits: List[dict],
    use_paraphrases: bool,
) -> List[List[Tuple[int, str, str]]]:
    edits = [x["edit"] for x in edits]
    examples: List[List[Tuple[int, str, str]]] = []
    for edit_id, edit in enumerate(edits):
        rewrite = edit["requested_rewrite"]
        prompt = rewrite["prompt"]
        subject = rewrite["subject"]
        target_new = rewrite["target_new"]
        # target_true = rewrite["target_true"]
        entity_id = edit_id  # TODO: fix this

        # neighbor_prompts = edit["neighbor_prompts"]
        # paraphrase_prompts = edit["paraphrase_prompts"]
        this_examples = []
        input_text = preprocess_input(prompt.format(subject))
        output_text = target_new["str"]
        this_examples = [(entity_id, input_text, output_text)]

        # Use the generation prompts as paraphrases for training
        if use_paraphrases:
            gen_prompts = edit["generation_prompts"]
            for gen_prompt in set(gen_prompts):
                input_text = preprocess_input(gen_prompt)
                this_examples.append((entity_id, input_text, output_text))
        examples.append(this_examples)
    return examples


class CounterfactDataset(Dataset, Sequence):
    def __init__(
        self, data_dir: Path, num_examples: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.examples = load_examples(data_dir, num_examples)

    def __getitem__(self, idx) -> dict:
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    src_dir = Path("../../data/counterfact")
    dataset = CounterfactDataset(src_dir, 1024)
    for i in range(800, 802):
        print(dataset[i]["input"])
