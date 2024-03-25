from typing import List, Callable

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from .editor import Editor
from .utils import gen_texts_qa


class FtEditor(Editor):
    """
    Baseline, standard finetuning a part of the model when applying an edit.
    """
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        train_params_getter: Callable,
        max_steps: int = 50,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps

        train_params = train_params_getter(model)
        self.optimizer = torch.optim.Adam(train_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=1.0)  # constant LR

    def edit_by_example(
        self,
        input_text: str,
        output_text: str,
    ):
        print("Editing by finetuning: {} -> {}".format(input_text, output_text))
        cur_step = 0
        input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to(
            self.model.device)
        labels = self.tokenizer(output_text, return_tensors='pt').input_ids.to(
            self.model.device)
        while cur_step < self.max_steps:
            outputs = self.model(input_ids=input_ids, labels=labels)

            # Backward
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Check if the output is correct
            cur_step += 1
            if cur_step >= self.max_steps:
                break
            gen_text = gen_texts_qa(self.model, self.tokenizer, [input_text])[0]
            if gen_text == output_text:
                # Correct prediction, early exit
                break

    def edit_by_examples(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ):
        for input_text, output_text in zip(input_texts, output_texts):
            self.edit_by_example(input_text, output_text)

    def gen_texts(self, input_text: List[str]) -> List[str]:
        if not input_text:
            return []
        return gen_texts_qa(self.model, self.tokenizer, input_text, max_length=20)
