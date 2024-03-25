from typing import List, Optional

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
from torch import nn

from .editor import Editor
from .baseless_serac import BaselessSerac, load_baseless_serac


class Serac(Editor, nn.Module):
    """
    SERAC model from (Eric Mitchell et al., 2022).

    Does not need a base model for evaluation. Need to be equal to nn.Module
    for moving to GPU.
    """
    def __init__(
        self,
        serac: BaselessSerac,
        base_model: Optional[T5ForConditionalGeneration] = None,
        base_tokenizer: Optional[T5TokenizerFast] = None,
    ):
        super().__init__()
        self.serac = serac
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer

    def edit_by_example(
        self,
        input_text: str,
        output_text: str,
        input_paraphrases: Optional[List[str]] = None,
    ):
        self.serac.edit(input_text, output_text)
        # If paraphrases, add them to the cache
        if input_paraphrases is not None:
            for paraphrase in input_paraphrases:
                self.serac.edit(paraphrase, output_text)

    def edit_by_examples(
        self,
        inputs: List[str],
        outputs: List[str],
        all_paraphrases: Optional[List[List[str]]] = None,
    ):
        # print(inputs)
        assert len(inputs) == len(outputs)
        if all_paraphrases is not None:
            assert len(all_paraphrases) == len(inputs)
            for para, output in zip(all_paraphrases, outputs):
                inputs += para
                outputs += [output] * len(para)
        self.serac.edit_many(inputs, outputs)

    def gen_texts_base_model(self, input_texts: List[str]) -> List[str]:
        return ["unanswerable"] * len(input_texts)
        # if self.flan_format:
        #     input_texts = [
        #         "Please answer this question: {}".format(inp) for inp in input_texts
        #     ]
        # inputs = self.tokenizer(
        #     input_texts,
        #     return_tensors="pt",
        #     padding="longest",
        # ).to(
        #     self.model.device
        # )
        # outputs = self.model.generate(**inputs)

        # if isinstance(outputs, Tensor):
        #     output_texts = self.tokenizer.batch_decode(
        #         outputs, skip_special_tokens=True
        #     )
        #     return output_texts
        # else:
        #     raise ValueError("Unexpected type of outputs.")

    def gen_text_cf(self, input_text: str, most_similar_idx: int) -> List[str]:
        return self.serac.generate_cf([input_text], [most_similar_idx])

    def gen_texts(self, input_texts: List[str]):
        if len(input_texts) == 0:
            return []
        if self.serac.is_cache_empty():
            return self.gen_texts_base_model(input_texts)
        # print(input_texts)
        # exit()
        sim_scores, most_similar_idxs = self.serac.forward_cls(input_texts)
        num_inputs = len(input_texts)

        # Need to evaluate each input separately during test generation.
        output_texts = []
        for input_idx in range(num_inputs):
            input_text = input_texts[input_idx]
            if sim_scores[input_idx] < 0.5:
                output = self.gen_texts_base_model([input_text])
                output_texts += output
            else:
                most_similar_idx = most_similar_idxs[input_idx]
                output_text = self.gen_text_cf(input_text, most_similar_idx)
                output_texts += output_text
        return output_texts


def load_serac(serac_ckpt_path: str, base_model_name: Optional[str] = None) -> Serac:
    """
    NOTE: We need base model to know whether the output of the counterfactual
    model is identical to that of the base model, but we can get the predictions
    separately.
    """
    baseless_serac = load_baseless_serac(serac_ckpt_path)
    if base_model_name is not None:
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        base_tokenizer = T5TokenizerFast.from_pretrained(base_model_name)
        if isinstance(base_model, T5ForConditionalGeneration):
            return Serac(baseless_serac, base_model, base_tokenizer)
        else:
            raise ValueError
    else:
        return Serac(baseless_serac)
