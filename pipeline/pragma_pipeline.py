"""KFP SDK v2 pipeline definition for PRAGMA pretraining.

Defines the end-to-end PRAGMA pretraining and evaluation pipeline
for deployment on OpenShift AI via Kubeflow Pipelines (KFP SDK v2).

Pipeline stages:
    1. preprocess_transactions — Tokenise and pack raw transaction data
    2. pretrain_pragma         — Run PRAGMA pretraining
    3. extract_embeddings      — Extract embeddings from trained model
    4. evaluate_downstream     — Evaluate on downstream tasks

The pipeline is compiled to YAML and registered via 00_register_pipeline.ipynb.
Compiled YAML is committed to the repository for GitOps deployment.

Deployment:
    The ArgoCD application (openshift/argocd/application.yaml) deploys
    the pipeline to OpenShift AI. See openshift/gitops/ for the full
    GitOps configuration.

Reference: Ostroukhov et al. (2026)
KFP SDK v2: https://www.kubeflow.org/docs/components/pipelines/
"""

from kfp import dsl

from .components_pragma import (
    evaluate_downstream,
    extract_embeddings,
    pretrain_pragma,
    preprocess_transactions,
)


@dsl.pipeline(
    name="pragma-pretraining-pipeline",
    description=(
        "End-to-end PRAGMA foundation model pretraining pipeline. "
        "Implements the masked event modelling objective from "
        "Ostroukhov et al. (2026), arXiv:2604.08649v1."
    ),
)
def pragma_pretraining_pipeline(
    raw_data_path: str,
    config_name: str = "pragma_s",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    mask_prob: float = 0.15,
    downstream_task: str = "fraud_detection",
) -> None:
    """Full PRAGMA pretraining and evaluation pipeline.

    Args:
        raw_data_path: S3/OBC path to raw transaction parquet files.
        config_name: Model size variant: 'pragma_s', 'pragma_m', 'pragma_l'.
        epochs: Number of pretraining epochs. Default: 10.
        batch_size: Per-device training batch size. Default: 32.
        learning_rate: AdamW learning rate. Default: 1e-4.
        mask_prob: Event masking probability. Default: 0.15.
        downstream_task: Task for evaluation. Default: 'fraud_detection'.
    """

    # Stage 1: Preprocess and tokenise transactions
    preprocess_op = preprocess_transactions(
        raw_data_path=raw_data_path,
    )

    # Stage 2: Pretrain PRAGMA with MEM objective
    pretrain_op = pretrain_pragma(
        train_dataset=preprocess_op.outputs["output_dataset"],
        config_name=config_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_prob=mask_prob,
    )

    # Stage 3: Extract embeddings from trained model
    extract_op = extract_embeddings(
        model=pretrain_op.outputs["output_model"],
        dataset=preprocess_op.outputs["output_dataset"],
    )

    # Stage 4: Evaluate on downstream task
    evaluate_op = evaluate_downstream(
        model=pretrain_op.outputs["output_model"],
        embeddings=extract_op.outputs["output_embeddings"],
        labelled_dataset=preprocess_op.outputs["output_dataset"],
        task_name=downstream_task,
    )


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(
        pipeline_func=pragma_pretraining_pipeline,
        package_path="pipeline/pragma_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/pragma_pipeline.yaml")
