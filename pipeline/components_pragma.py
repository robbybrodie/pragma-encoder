"""KFP SDK v2 pipeline components for PRAGMA training.

Defines Kubeflow Pipelines (KFP SDK v2) components for the PRAGMA
pretraining and evaluation pipeline on OpenShift AI.

Components follow the KFP SDK v2 decorator pattern consistent with
the adjacent transaction-foundation-model-openshiftai/ pipeline.

Components defined here:
    preprocess_transactions — Tokenise and pack raw transaction data
    pretrain_pragma         — Run PRAGMA pretraining (KFTO PyTorchJob)
    extract_embeddings      — Extract embeddings from trained model
    evaluate_downstream     — Run downstream task evaluation

Deployment note:
    These components are compiled to YAML by 00_register_pipeline.ipynb
    and deployed to the OpenShift AI pipeline server. The compiled YAML
    is committed intentionally (see .gitignore for pipeline/*.yaml policy).

Reference: Ostroukhov et al. (2026)
KFP SDK v2: https://www.kubeflow.org/docs/components/pipelines/
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="pragma-encoder:latest",
    packages_to_install=[],
)
def preprocess_transactions(
    raw_data_path: str,
    output_dataset: Output[Dataset],
    n_buckets: int = 100,
    bpe_vocab_size: int = 8000,
    max_history_len: int = 512,
) -> None:
    """Tokenise and pack raw transaction data for PRAGMA pretraining.

    Applies the FinancialTokenizerPipeline to raw transaction records
    and packs sequences for training efficiency (Section 2.4).

    Args:
        raw_data_path: Path to raw transaction parquet files.
        output_dataset: KFP output dataset artifact.
        n_buckets: Percentile buckets for numerical fields. Default: 100.
        bpe_vocab_size: BPE vocabulary size for text fields. Default: 8000.
        max_history_len: Maximum events per customer history. Default: 512.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing transactions from {raw_data_path}")
    logger.info("Placeholder — implement tokenisation and packing.")
    # TODO: Implement full preprocessing pipeline


@dsl.component(
    base_image="pragma-encoder:latest",
)
def pretrain_pragma(
    train_dataset: Input[Dataset],
    config_name: str,
    output_model: Output[Model],
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    mask_prob: float = 0.15,
) -> None:
    """Run PRAGMA pretraining using the masked event modelling objective.

    Submits a KFTO PyTorchJob to OpenShift AI for distributed training.
    Single-node training runs locally within the component.

    Args:
        train_dataset: Tokenised training data from preprocess_transactions.
        config_name: Model size: 'pragma_s', 'pragma_m', or 'pragma_l'.
        output_model: KFP output model artifact (checkpoint path).
        epochs: Number of training epochs. Default: 10.
        batch_size: Per-device batch size. Default: 32.
        learning_rate: AdamW learning rate. Default: 1e-4.
        mask_prob: Masking probability for MLM. Default: 0.15.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Pretraining PRAGMA ({config_name}) for {epochs} epochs")
    logger.info("Placeholder — implement training loop with NeMo AutoModel.")
    # TODO: Implement training with NeMo AutoModel and KFTO PyTorchJob


@dsl.component(
    base_image="pragma-encoder:latest",
)
def extract_embeddings(
    model: Input[Model],
    dataset: Input[Dataset],
    output_embeddings: Output[Dataset],
    pooling: str = "hist",
    batch_size: int = 64,
) -> None:
    """Extract PRAGMA embeddings from a trained model.

    Args:
        model: Trained PRAGMA model checkpoint.
        dataset: Input transaction dataset.
        output_embeddings: Output embedding dataset (numpy arrays).
        pooling: Pooling strategy: 'hist', 'mean', or 'last'. Default: 'hist'.
        batch_size: Inference batch size. Default: 64.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting embeddings with pooling={pooling}")
    logger.info("Placeholder — implement embedding extraction.")
    # TODO: Implement embedding extraction


@dsl.component(
    base_image="pragma-encoder:latest",
)
def evaluate_downstream(
    model: Input[Model],
    embeddings: Input[Dataset],
    labelled_dataset: Input[Dataset],
    task_name: str,
    output_metrics: Output[Dataset],
) -> None:
    """Evaluate PRAGMA embeddings on downstream tasks.

    Args:
        model: Trained PRAGMA model checkpoint.
        embeddings: Pre-extracted embeddings (optional shortcut).
        labelled_dataset: Labelled dataset for the downstream task.
        task_name: Task name: 'fraud_detection', 'churn', 'credit_risk'.
        output_metrics: Output metrics JSON artifact.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating downstream task: {task_name}")
    logger.info("Placeholder — implement downstream evaluation.")
    # TODO: Implement downstream evaluation with DownstreamEvaluator
