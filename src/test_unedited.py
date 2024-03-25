from pathlib import Path

from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch

from args import Args
from test_utils import evaluate
from utils import set_seed, get_data_name_and_task_type
from data_utils import load_data

from model.editor import Editor
from model.utils import gen_texts_qa, gen_texts_bool_qa


device = "cuda:1" if torch.cuda.is_available() else "cpu"


class Unedited(Editor):
    def __init__(self, model, tokenizer, task_type):
        self.model = model
        self.tokenizer = tokenizer
        self.task_type = task_type

    def gen_texts(self, input_texts):
        if len(input_texts) == 0:
            return []
        if self.task_type == "nli":
            output_texts = gen_texts_bool_qa(self.model, self.tokenizer, input_texts)
        elif self.task_type == "mrc":
            output_texts = gen_texts_qa(self.model, self.tokenizer, input_texts)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return output_texts


def load_model(pretrained_name, task_type):
    print("Loading model and tokenizer...")
    # NOTE: 4GB VRAM cannot do tinference with T5-Large
    model = T5ForConditionalGeneration.from_pretrained(
        args.pretrained_name, local_files_only=True
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        args.pretrained_name, local_files_only=True
    )
    model = model.to(device)
    editor = Unedited(model, tokenizer, task_type)
    return editor


if __name__ == "__main__":
    args = Args().parse_args()
    print(args)
    set_seed(0)
    data_path = Path(args.data_path)
    data_name, task_type = get_data_name_and_task_type(args)
    # print(task_type)
    output_dir = Path("result", data_name, "unedited", args.pretrained_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.save(str(output_dir / "args.json"))

    model = load_model(args.pretrained_name, task_type)

    # Data
    print("Loading data")
    examples = load_data(
        data_path,
        args.num_examples,
        bool(args.nli_with_mrc),
        convert_to_bool_q=bool(args.mrc_convert_to_bool_q),
    )
    print(f"Loaded {len(examples)} examples.")

    # Evaluate
    test_output_dir = output_dir / "test"
    test_output_dir.mkdir(exist_ok=True, parents=True)
    result = evaluate(
        model,
        dataset=examples,
        output_dir=test_output_dir,
    )
