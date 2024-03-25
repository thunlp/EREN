from pathlib import Path

import torch

from utils import set_seed, get_data_name_and_task_type
from data_utils import load_data
from data_classes.utils import dump_json
from model.serac import load_serac
from args import Args
from test_utils import apply_edits, evaluate


def get_task_type(data_name):
    if "fever" in data_name:
        return "nli"
    if "cf" in data_name:
        return "mrc"
    raise ValueError(f"Unknown data name: {data_name}")


def edit(model, dataset, data_name):
    if "fever" in data_name:
        # For FEVER, we will just apply the first `num_examples` examples
        # as edits, and use another `num_examples` examples as a measure
        # for drawdown (which are easy out-of-scope examples).
        assert len(dataset) % 2 == 0
        apply_edits(model, dataset[: len(dataset) // 2], edit_by_example=True)
    elif "cf" in data_name:
        apply_edits(model, dataset, edit_by_example=True)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")


def main():
    args = Args().parse_args()
    print(args)
    set_seed(0)
    data_path = Path(args.data_path)
    data_name, task_type = get_data_name_and_task_type(args)
    ckpt_name = Path(args.serac_ckpt_path).stem
    output_dir = Path("result", data_name, "serac", ckpt_name, str(args.num_examples))
    output_dir.mkdir(parents=True, exist_ok=True)
    args.save(str(output_dir / "args.json"))

    print("Loading data")
    dataset = load_data(
        data_path,
        args.num_examples,
        convert_fc_to_questions="fever" in data_name,
        convert_to_bool_q=(args.mrc_convert_to_bool_q),
    )
    print(f"Loaded {len(dataset)} examples")

    model = load_serac(args.serac_ckpt_path)
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model.to(device)

    edit(model, dataset, data_name)

    # Save cache for visualization
    dump_json(model.serac.cache_inputs, output_dir / "cache_inputs.json")
    dump_json(model.serac.cache_outputs, output_dir / "cache_outputs.json")
    evaluate(
        model,
        dataset,
        output_dir,
    )


if __name__ == "__main__":
    main()
