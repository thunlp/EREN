from pathlib import Path

from args import Args
from data_utils import dump_json, load_data
from model.eren import Eren, load_eren_contriever
from utils import set_seed, get_data_name_and_task_type
from test_utils import apply_edits, evaluate


def load_model(args: Args, data_name, task_type: str) -> Eren:
    """data_name and task_name will determine what options are fed to the model."""
    # Create options
    if "cf" in data_name:
        ans_options = None
    elif "fever" in data_name:
        if task_type == "mrc":
            # options = ["yes", "unanswerable", "no"]
            ans_options = ["Yes", "It's impossible to say", "No"]
        elif task_type == "nli":
            ans_options = ["Yes", "It's impossible to say", "No"]
        else:
            raise ValueError(f"Invalid task type: {task_type}")
    elif "hotpot" in data_name:
        ans_options = None
    else:
        raise ValueError(f"Invalid data name: {data_name}")
    model = load_eren_contriever(
        args.pretrained_name,
        task_type=task_type,
        ans_options=ans_options,
        one_step_mrc=bool(args.one_step_mrc),
        num_context_examples=args.num_context_examples,
    )
    model = model.to(args.device)
    return model


def edit_model(model, dataset, data_name):
    if "fever" in data_name:
        # In FEVER, we use the later half examples as out-of-scope examples
        assert len(dataset) % 2 == 0
        num_inscope = len(dataset) // 2
        apply_edits(model, dataset[:num_inscope], edit_by_statement=True)
    elif "cf" in data_name:
        apply_edits(model, dataset, edit_by_statement=True)
    elif "hotpot" in data_name:
        num_inscope = len(dataset) // 2
        apply_edits(model, dataset[:num_inscope], edit_by_statement=True)
    else:
        raise ValueError(f"Unknown data name: {data_name}")


if __name__ == "__main__":
    args = Args().parse_args()
    print(args)
    data_path = Path(args.data_path)
    data_name, task_type = get_data_name_and_task_type(args)
    output_dir = Path(
        "result",
        data_name,
        "emoren-onestep" if bool(args.one_step_mrc) else "emoren",
        args.pretrained_name,
        # Path(args.serac_ckpt_path).stem,
        f"contriever-top{args.num_context_examples}",
        str(args.num_examples),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args.save(str(output_dir / "args.json"))

    # Model
    print("Loading model and tokenizer")

    # Data
    print("Loading data")
    dataset = load_data(
        data_path,
        args.num_examples,
        convert_fc_to_questions=bool(args.nli_with_mrc),
        convert_to_bool_q=bool(args.mrc_convert_to_bool_q),
    )
    print("Num examples:", len(dataset))
    set_seed(0)

    # Model
    model = load_model(args, data_name, task_type)
    edit_model(model, dataset, data_name)
    assert sum([args.mrc_convert_to_bool_q, args.nli_with_mrc]) <= 1

    # Run
    dump_json(model.notes, output_dir / "notes.json")
    evaluate(
        model,
        dataset,
        output_dir / "test",
    )
    print("Result dumped to", output_dir)
    print("Done")
