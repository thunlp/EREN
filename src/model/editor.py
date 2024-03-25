from typing import List, Optional


class Editor:
    """
    Model Editor interface.

    This does not require base model and tokenizer because for evaluation,
    we may not need that as long as we know that the final output will be
    identical to that of the base model.
    """
    def edit_by_statements(self, statements: List[str]):
        """Apply edits using a declarative statements."""
        raise NotImplementedError

    def edit_by_examples(
        self,
        input_text: List[str],
        output_text: List[str],
        subj_id: Optional[List[str]] = None,
        paraphrases: Optional[List[List[str]]] = None,
    ):
        """
        Most existing works apply edits using input-output pairs.

        Some method might use subj_id (subject entity ID).
        """
        raise NotImplementedError

    def gen_texts(
        self,
        input_text: List[str],
        subj_id: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Generate texts from input texts.
        subj_id is required for some model editors.
        """
        raise NotImplementedError
