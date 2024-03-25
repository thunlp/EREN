from typing import List, Tuple, Dict, Optional

from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch import nn
import torch

from test_utils import gen_texts


def prepend_to_texts(
    input_texts: List[str], context: str, flan_format: bool
) -> List[str]:
    prepended = []
    for i in range(len(input_texts)):
        if context != "":
            prepended.append(context + " Q:" + input_texts[i] + " A:")
        else:
            prepended.append("Q: " + input_texts[i] + " A:")
    return prepended


def create_context(examples: list, flan_format: bool) -> str:
    # TODO: Support non-FLAN format
    if not examples:
        return ""
    example_strs = ["Q: " + ex[0] + " A: " + ex[1] for ex in examples]
    return " ".join(example_strs)


class IncontextEL(nn.Module):
    """
    Baseline for in-context learning, where demonstrations are retrieved using
    entity linking (EL).
    """

    def __init__(self, model, tokenizer, flan_format: bool):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.flan_format = flan_format

        self.context_examples: Dict[str, List[Tuple[str, str]]] = {}

    def edit_by_example(
        self, input_text: str, output_text: str, subj_id: str, paraphrases: List[str]
    ):
        """
        This will pre-compute context for the subject of the example.
        """
        # Create context using training examples
        train_inputs: List[str] = [input_text] + paraphrases
        context_examples = [(inp, output_text) for inp in train_inputs]
        assert len(train_inputs) > 0
        # context = create_context(context_examples, self.flan_format)
        # assert "Q: Q:" not in context, context + " | " + str(train_examples)
        self.context_examples[subj_id] = context_examples

    def gen_texts(
        self,
        input_texts: List[str],
        num_context_examples: Optional[int] = None,
        subj_id: Optional[str] = None,
    ) -> List[str]:
        with torch.no_grad():
            context_examples = []
            if subj_id is not None:
                context_examples = self.context_examples[subj_id]
                if context_examples is not None:
                    context_examples = context_examples[:num_context_examples]
            context = create_context(context_examples, self.flan_format)
            input_texts = prepend_to_texts(input_texts, context, self.flan_format)
            # NOTE: Passing flan_format=False to gen_text because
            # creating the prompt needs to be handled when prepending the
            # context.
            result = gen_texts(
                self.model, self.tokenizer, input_texts, flan_format=False
            )
            return result


def load_incontext_el(
    pretrained_name: str,
    cache_dir: str,
) -> IncontextEL:
    print("Loading tokenizer and model from:", pretrained_name)
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_name,
        cache_dir=cache_dir,
    )
    if isinstance(model, T5ForConditionalGeneration) and isinstance(
        tokenizer, T5TokenizerFast
    ):
        return IncontextEL(model, tokenizer, flan_format="flan" in pretrained_name)
    else:
        raise ValueError
