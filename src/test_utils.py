from typing import List, Dict, Union
from pathlib import Path
import time

from model.editor import Editor
from data_classes.utils import dump_json


def evaluate_same_subject(
    model: Editor,
    examples_same_subj: List[str],
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Perform evaluation on examples on same subject but different relation.
    
    `examples_same_subj`: list of input texts.
    """
    results: List[dict] = []
    for example in examples_same_subj:
        input_texts = example["inputs"]
        preds = model.gen_texts(input_texts)
        this_result = {
            "inputs": input_texts,
            "preds": preds,
        }
        results.append(this_result)
    return results


def get_statements(dataset: list) -> List[str]:
    all_statements = []
    for example in dataset:
        statements = example["edit_statement"]
        if isinstance(statements, str):
            statements = [statements]
        for i, statement in enumerate(statements):
            if statement.endswith("."):
                statements[i] = statement[:-1]
        all_statements += statements
    return all_statements


def get_edit_examples(dataset: list):
    pairs = [eg["edit_scope"][0] for eg in dataset if eg["edit_scope"]]
    return list(zip(*pairs))


def apply_edits(
    model: Editor,
    dataset: list,
    edit_by_statement: bool = False,
    edit_by_example: bool = False,
):
    assert (
        sum([edit_by_statement, edit_by_example]) == 1
    ), "Must choose one of edit_by_statements or edit_by_example"
    # Make all edits
    print("=== Editing model ===")
    if edit_by_statement:
        statements = get_statements(dataset)
        model.edit_by_statements(statements)
    elif edit_by_example:
        inputs, outputs = get_edit_examples(dataset)
        # print(inputs)
        model.edit_by_examples(inputs, outputs)
    else:
        raise ValueError("Must choose one of edit_by_statements or edit_by_example")


def evaluate(
    model: Editor,
    dataset: list,
    output_dir: Path,
    process_input_texts: callable = None,
):
    """
    :param task_type: This is only passed to the model.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    print(output_dir)
    print("=== Evaluating ===")
    all_outputs = []
    start_time = time.time()
    for step, example in enumerate(dataset):
        ex_id = example["id"]
        edit_statement = example["edit_statement"]
        time_elapsed = round(time.time() - start_time)
        print(
            {
                "time": time_elapsed,
                "step": step,
                "id": ex_id,
                "edit": edit_statement,
            }
        )

        this_outputs = {}
        # In-scope examples
        edit_scope = example["edit_scope"]
        input_texts = [x[0] for x in edit_scope]
        label_texts = [x[1] for x in edit_scope]
        if process_input_texts is not None:
            input_texts = process_input_texts(input_texts)
        output_texts = model.gen_texts(input_texts)
        this_outputs["edit_scope"] = output_texts

        # Out-of-scope examples (FEVER does not have this)
        if "unrelated" in example:
            this_outputs["unrelated"] = {}
            for key in example["unrelated"]:
                input_texts = example["unrelated"][key]
                pred_texts = model.gen_texts(input_texts)
                this_outputs["unrelated"][key] = pred_texts

        # Dump temporary result
        all_outputs.append({
            "id": example["id"],
            "edit_statement": edit_statement,
            "targets": label_texts,
            "outputs": this_outputs,
        })
        if step % 10 == 0:
            dump_json(all_outputs, output_dir / "preds.json")
    dump_json(all_outputs, output_dir / "preds.json")
    print(f"preds dumped to {output_dir / 'preds.json'}")
