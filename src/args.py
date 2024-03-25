from tap import Tap


class Args(Tap):
    """
    Use for both training and testing, but some parameters are ignored
    during testing.
    """

    # Output args
    output_dir = "result"
    seed: int = 1
    device: str = 'cuda'

    # Data args
    data_path: str = "../data/cfh_filtered.json"
    num_examples: int = 1
    one_step_mrc: int = 0

    # Whether to change statements into boolean question.
    mrc_convert_to_bool_q: int = 0

    # Whether to reformat NLI tasks to MRC.
    nli_with_mrc: int = 0

    pretrained_name: str = "google/flan-t5-xl"
    """Name for the base model"""

    # Serac
    serac_ckpt_path: str = "../ckpts/serac-no_da.pt"
    """Path for the SERAC checkpoint trained with its official repo."""

    # In-context
    num_context_examples: int = 5
    """Number of examples to use for in-context learning."""

    # train_params: str = "mlp.enc.5"
    train_params: str = "all"
    """
    Which parameters to train:
    - "all" for full finetuning
    - "mlp.enc.x" for tuning MLP of the x'th layer.
    """

    lr: float = 1e-5  # Learning rate
    num_epochs: int = 200
    """Maximum number of update steps when training the soft prompts."""

    early_exit = 1
    """If 1, training stops when the model produces the target output
    on all evaluation examples."""

    log_interval: int = 10
    eval_interval: int = 10
    """Number of steps between evaluations. This will affect early exit
    because exit only happens after evaluation."""

    mode: str = "train_test"
    """"test" / "train" for test or train only"""
