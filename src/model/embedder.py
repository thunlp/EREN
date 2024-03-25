from typing import List

from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch


class Embedder:
    def embed(self, input_texts: List[str]):
        raise NotImplementedError


class Contriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            'facebook/contriever')
        self.model = AutoModel.from_pretrained(
            'facebook/contriever').to('cuda')
        self.model.eval()
    def mean_pooling(self, token_embeds, mask):
        token_embeds = token_embeds.masked_fill(~mask[..., None].bool(), 0.)
        sent_embeds = token_embeds.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sent_embeds
    def embed(self, sents: List[str]) -> Tensor:
        inputs = self.tokenizer(
            sents, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeds = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return embeds


if __name__ == "__main__":
    model = Contriever()

    sents = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie,"
        " a doctor of French Catholic origin from Alsace."
    ]

    embeds = model.embed(sents)
    sim_score = embeds @ embeds.T
    print(sim_score)
