from pathlib import Path

from torch.utils.data import Dataset

if __name__ == "__main__":
    from utils import iter_jsonl
else:
    from .utils import iter_jsonl


def load_examples(data_file, num_examples: int):
    examples = []
    for eg in iter_jsonl(data_file, num_examples):
        assert len(eg["alternatives"]) == 1
        examples.append(
            {
                "id": eg["id"],
                "input": eg["input"],
                "output_true": eg["output"][0]["answer"],
                "output_new": eg["alternatives"][0],
                "paraphrases": eg["filtered_rephrases"],
            }
        )
    return examples


class FeverDataset(Dataset):
    def __init__(
        self, data_dir: Path, num_examples: int, data_type: str = "validation"
    ):
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.examples = load_examples(data_dir / "fever-dev-kilt.jsonl", num_examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


if __name__ == "__main__":
    data_dir = Path("../../data/fever")
    dataset = FeverDataset(data_dir, 32)
    questions = [eg["input"] for eg in dataset]
    print(questions)
