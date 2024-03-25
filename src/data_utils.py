from pathlib import Path
from typing import List, Optional
import random
import json


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


def remove_ending_period(text: str):
    if text.endswith("."):
        text = text[:-1]
    return text


def convert_statement_to_yn_question(texts: List[str]) -> List[str]:
    """Convert a declarative sentence into a yes/no question"""
    texts = [remove_ending_period(t) for t in texts]
    return ["Is it true that " + t + "?" for t in texts]


def load_data(
    data_path: Path,
    num_examples: int,
    convert_fc_to_questions: bool,
    convert_to_bool_q: bool,
):
    """
    Load data from `data_path`, then randomly sample `num_examples` from the data.

    params:
    - data_path: Path to the data file.
    - num_examples: Number of examples to load.
    - convert_to_questions: Whether to convert inputs in fever to boolean question.
    - convert_to_bool_q: Whether to use edit_statements to convert to boolean questions.
    (declarative statements) to questions.
    """
    data_name = data_path.stem
    dataset = load_json(data_path)

    # Sanity checks
    if convert_to_bool_q:
        assert "cf" in data_name
    if convert_fc_to_questions:
        assert "fever" in data_name

    if "fever" in data_name:
        # For FEVER, we will just apply the first `num_examples` examples
        # as edits, and use another `num_examples` examples as a measure
        # for drawdown (which are easy out-of-scope examples).
        assert len(dataset) >= 2 * num_examples
        dataset = random.sample(dataset, 2 * num_examples)

        # Preprocess inputs
        if convert_fc_to_questions:
            for eg in dataset:
                input_texts = [x[0] for x in eg["edit_scope"]]
                input_texts = convert_statement_to_yn_question(input_texts)
                for i, t in enumerate(input_texts):
                    eg["edit_scope"][i][0] = t
    elif "cf" in data_name:
        dataset = random.sample(dataset, num_examples)

        if convert_to_bool_q:
            # Replace input with a question converted from edit statement
            # Delete all other input questions (in-scope and out-of-scope)
            for eg in dataset:
                edit_statement = eg["edit_statement"]
                if edit_statement.endswith("."):
                    edit_statement = edit_statement[:-1]
                # eg["input"] = [edit_statement + ", is that right?"]
                eg["edit_scope_true_ans"] = ["no"]
                eg["edit_scope"] = [
                    eg["edit_scope"][0],
                    (f"Is it true that {edit_statement}?", "yes"),
                ]
                for key in eg["unrelated"]:
                    eg["unrelated"][key] = []
    elif "hotpot" in data_name:
        assert len(dataset) >= 2 * num_examples
        # dataset = dataset[: 2 * num_examples]
        dataset = random.sample(dataset, 2 * num_examples)
        for eg in dataset:
            for i, es in enumerate(eg["edit_statement"]):
                if es.endswith('.'):
                    eg["edit_statement"][i] = es[:-1]
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    return dataset
