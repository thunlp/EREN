"""
Code for running baseline experiments. Have 4 different models:

- Full finetuning
- Finetune only one layer
- Finetune only one MLP layer
- Finetune only one attention layer
- Finetune while constraining weight updates with a L1 norm.

> in all experiments, we stop training when the model produces the target
output.
"""

from pathlib import Path

from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

from args import Args
from data_utils import load_data
from model.ft import FtEditor
from test_utils import evaluate, apply_edits
from utils import get_data_name_and_task_type, set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(args: Args) -> FtEditor:
    pretrained_name = args.pretrained_name

    def get_train_params(model):
        '''Passed to the editor to select with parameters to finetune.'''
        if args.train_params == "all":
            return model.parameters()
        elif args.train_params.startswith("mlp"):
            layer = int(args.train_params.split('.')[-1])
            if "enc" in args.train_params:
                params = []
                for name, p in model.named_parameters():
                    if name.startswith(f"encoder.block.{layer}.layer.1"):
                        params.append(p)
                return params
            elif "dec" in args.train_params:
                params = []
                for name, p in model.named_parameters():
                    if name.startswith(f"decoder.block.{layer}.layer.2"):
                        params.append(p)
                return params
        else:
            raise ValueError(f"Invalid train_params: {args.train_params}")

    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_name).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_name)
    editor = FtEditor(model, tokenizer, lr=args.lr, max_steps=args.num_epochs, train_params_getter=get_train_params)
    return editor


def main():
    args = Args().parse_args()
    print(args)

    data_path = Path(args.data_path)
    data_name, task_type = get_data_name_and_task_type(args)
    output_dir = Path(
        "result",
        data_name,
        "ft",
        args.pretrained_name,
        args.train_params,
        str(args.num_examples),
        f'seed{args.seed}',
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json"))

    # Data
    set_seed(args.seed)
    dataset = load_data(
        data_path,
        args.num_examples,
        convert_fc_to_questions="fever" in data_name,
        convert_to_bool_q=bool(args.mrc_convert_to_bool_q),
    )
    print(f"Loaded {len(dataset)} examples")

    # Load model
    print("Loading model...")
    model = load_model(args)

    # Test
    apply_edits(model, dataset[:args.num_examples], edit_by_example=True)
    evaluate(model, dataset, output_dir / "test")
    print("Result dumped to", output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
