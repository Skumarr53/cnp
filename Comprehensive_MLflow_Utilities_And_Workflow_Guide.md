
# MLflow Custom Module Documentation

## Overview
This documentation provides a structured, user-friendly overview of the custom MLflow module. It aims to help non-technical users understand how models are trained, tracked, and transitioned into production using MLflow. All key code elements have been summarized and explained with a focus on clarity and simplicity.

## Table of Contents
- [MLflow Concepts: Experiments vs Model Registry](#mlflow-concepts-experiments-vs-model-registry)
  - [MLflow Experiment](#mlflow-experiment)
  - [MLflow Model Registry](#mlflow-model-registry)
  - [Key Differences](#key-differences-between-mlflow-experiments-and-model-registry)
- [Script Paths](#script-paths)
- [Naming Conventions](#naming-conventions)
- [Setting up and Running Experiments](#setting-up-and-running-experiments)
  - [Initializing the `ExperimentManager`](#initializing-the-experimentmanager)
  - [Running Experiments and Creating Runs](#running-experiments-and-creating-runs)
- [Model Training Steps](#model-training-steps)
- [Model Evaluation](#model-evaluation)
- [Selecting the Best Model](#selecting-the-best-model)
- [Model Registration and Transitioning to Production](#model-registration-and-transitioning-to-production)
  - [Initializing the `ModelTransition` Class](#initializing-the-modeltransition-class)
  - [Transitioning Models](#transitioning-models)
- [Arguments and Function Calls](#arguments-and-function-calls)
- [Summary](#summary)

---

## MLflow Concepts: Experiments vs Model Registry

### MLflow Experiment
An **MLflow Experiment** is a logical container used to group multiple runs together. Each run typically corresponds to a single execution of the training process with a given configuration. Experiments help you organize and compare different modeling approaches, parameter sets, and datasets. In short, an experiment:
- Tracks multiple model runs.
- Stores metrics, parameters, and artifacts (such as model outputs and logs).
- Simplifies comparison of models trained under different conditions.

### MLflow Model Registry
The **MLflow Model Registry** is a centralized hub where trained models can be registered, versioned, and assigned to different stages such as "Staging" or "Production." By using the Model Registry:
- You can have a named entry (model name) under which multiple versions of the model are recorded.
- Facilitate easy rollbacks, promotions, and the clear staging of models (e.g., Development → Staging → Production).
- Make model promotion and governance processes more systematic and transparent.

### Key Differences Between MLflow Experiments and Model Registry
| Aspect              | MLflow Experiment                                   | MLflow Model Registry                                                |
|---------------------|-----------------------------------------------------|----------------------------------------------------------------------|
| Purpose             | Group and track training runs, metrics, and params  | Store, version, and manage production-ready models                    |
| Entities Managed    | Runs (metrics, parameters, artifacts)               | Model versions (registered models with version history)               |
| Scope               | Development and experimentation phases              | Deployment, staging, and productionization phases                     |
| Access Pattern       | Used primarily during training and evaluation phases | Accessed when promoting models to production or managing versions     |

---

## Script Paths
All referenced scripts are located in the `centralized_nlp_package` directory, making it easier to navigate from the project’s root. Key script locations include:

- Experiment Management Utilities:  
  `centralized_nlp_package/mlflow_utils/experiment_manager.py`

- Model Selection Utilities:  
  `centralized_nlp_package/mlflow_utils/model_selector.py`

- Model Transition Utilities:  
  `centralized_nlp_package/mlflow_utils/model_transition.py`

Updating paths ensures a direct mapping to code files and allows users to click through from documentation (e.g., GitHub) to code in a single step.

---

## Naming Conventions

**Entities Involved:**  
- **Experiment Name:** Groups a series of runs related to a single large-scale experiment.  
- **Run Name:** Identifies an individual model training run within an experiment.

**Experiment Name Structure:**
```
/Users/{user_id}/{base_name}_{data_src}_{run_date}
```
- **`user_id`:** Identifier for the user.
- **`base_name`:** A descriptive name for the experiment (e.g., the domain or approach used).
- **`data_src`:** Indicates which data source or dataset is involved.
- **`run_date`:** The date when the experiment was initiated (format: YYYYMMDD).

**Example:**
If `user_id = "santhosh.kumar3@voya.com"`, `base_name = "Mass_ConsumerTopic_FineTune_DeBERTa_v3"`, and `data_src = "CallTranscript"` on `20241205`,  
the `experiment_name` would be:
```
/Users/santhosh.kumar3@voya.com/Mass_ConsumerTopic_FineTune_DeBERTa_v3_CallTranscript_20241205
```

**Run Name Structure:**
```
{base_model_name}_{dataset_name}_param_set{index}
```
- **`base_model_name`:** Extracted from the base model path (e.g., `deberta-v3-base-zeroshot-v2`).
- **`dataset_name`:** Derived from the dataset version filename (e.g., `full_dataset.csv` → `full_dataset`).
- **`index`:** Parameter set number (e.g., `param_set1`).

---

## Setting up and Running Experiments

### Initializing the `ExperimentManager`
**Location:** `centralized_nlp_package/mlflow_utils/experiment_manager.py`

To initiate experiments, you create an instance of `ExperimentManager`:
```python
from centralized_nlp_package.mlflow_utils.experiment_manager import ExperimentManager

experiment_manager = ExperimentManager(
    base_name="Mass_ConsumerTopic_FineTune_DeBERTa_v3",
    data_src="CallTranscript",
    dataset_versions=["full_dataset.csv", "train_sample.csv"],
    hyperparameters=[
        {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16},
        {"n_epochs": 8, "learning_rate": 3e-5, "weight_decay": 0.02, "train_batch_size": 24, "eval_batch_size": 24}
    ],
    base_model_versions=[
        "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-base-zeroshot-v2",
        "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"
    ],
    output_dir="/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/trained_models/",
    train_file="/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/{data_version}",
    validation_file="/Workspace/Users/santhosh.kumar3@voya.com/MLFlow_and_NLI_finetune/data/test.csv",
    evalute_pretrained_model=True,
    user_id="santhosh.kumar3@voya.com"
)
```

**Parameters Explained:**
- `base_name`: The core name or theme of the experiment (e.g., a project or model type).
- `data_src`: Identifies the data source (e.g., "CallTranscript").
- `dataset_versions`: A list of dataset file names. Each dataset leads to separate runs.
- `hyperparameters`: A list of dictionaries, each defining a unique configuration of training parameters.
- `base_model_versions`: A list of base model paths from which you fine-tune.
- `output_dir`: Directory to save trained models and outputs.
- `train_file`, `validation_file`: File patterns/paths for training and validation datasets.
- `evalute_pretrained_model`: If `True`, evaluates the base model before fine-tuning.
- `user_id`: Identifies the user to help form the experiment name.

### Running Experiments and Creating Runs
Once the `ExperimentManager` is set up, start the runs:
```python
experiment_manager.run_experiments()
```
This:
- Sets the MLflow experiment name (based on parameters given).
- Evaluates the pretrained model if `evalute_pretrained_model` is `True`.
- Iterates over each `base_model` and each `dataset_version`.
- For each combination and `hyperparameter` set, a new MLflow run is created.
- Each run logs parameters, metrics, and the resulting trained model artifact to MLflow.

---

## Model Training Steps
During each run:
- Training is performed via a `model.train()` call on the selected dataset.
- Hyperparameters (e.g., `n_epochs`, `learning_rate`) are passed to this training method.
- After training, `mlflow.transformers.log_model()` is used to log the trained model artifacts.

---

## Model Evaluation
**Pretrained Model Evaluation:**  
If `evalute_pretrained_model` is `True`, the baseline (unfined-tuned) model is evaluated using the validation data. This helps compare the improvement gained by fine-tuning.

**Evaluation Metrics:**
- Accuracy
- F1-Score
- Precision
- Recall
- ROC-AUC

Metrics are automatically logged to MLflow for each run, making it easy to compare models and identify the best performing configurations.

---

## Selecting the Best Model
**Class:** `ModelSelector`  
**Location:** `centralized_nlp_package/mlflow_utils/model_selector.py`

To find top models:
```python
from centralized_nlp_package.mlflow_utils.model_selector import ModelSelector

selector = ModelSelector(
    experiment_name="/Users/santhosh.kumar3@voya.com/Mass_ConsumerTopic_FineTune_DeBERTa_v3_CallTranscript_20241205",
    metric="accuracy"
)
```

**Methods:**
- `selector.get_best_model()`: Returns the run with the highest accuracy.
- `selector.get_best_models_by_tag(tag)`: Groups runs by a tag (e.g., `base_model_name`) and returns the best run from each group.
- `selector.get_best_models_by_param(param)`: Similar grouping but by parameter.

**Example Usage:**
```python
best_run = selector.get_best_model()
best_models_by_base_model = selector.get_best_models_by_tag("base_model_name")
```

---

## Model Registration and Transitioning to Production

### Initializing the `ModelTransition` Class
**Location:** `centralized_nlp_package/mlflow_utils/model_transition.py`

```python
from centralized_nlp_package.mlflow_utils.model_transition import ModelTransition

model_name = "Topic_Modeling_Consumer_Topic"
transitioner = ModelTransition(model_name=model_name, registry_uri="databricks")
```

**Parameters:**
- `model_name`: The name under which the model is registered in the Model Registry.
- `registry_uri`: URI to the Model Registry (e.g., "databricks").

### Transitioning Models
If you have a best run and want to register and promote it to Production:
```python
transitioner.transition_model(
    stage="Production",
    experiment_name="/Users/santhosh.kumar3@voya.com/Mass_ConsumerTopic_FineTune_DeBERTa_v3_CallTranscript_20241205",
    metric="accuracy"
)
```

This:
- Finds the best model run from the given experiment.
- Registers that run’s model in the Model Registry under `model_name`.
- Moves it to the specified stage (e.g., `Production`).

---

## Arguments and Function Calls

**`ExperimentManager()` Arguments:**
```python
ExperimentManager(
    base_name="...",
    data_src="...",
    dataset_versions=["..."],
    hyperparameters=[{...}, {...}],
    base_model_versions=["..."],
    output_dir="...",
    train_file="...",
    validation_file="...",
    evalute_pretrained_model=True/False,
    user_id="..."
)
```

**`run_experiments()` call:**
```python
experiment_manager.run_experiments()
```
No direct arguments; it uses the instance’s initialized parameters.

**`ModelSelector()` Arguments:**
```python
ModelSelector(
    experiment_name="...",
    metric="accuracy"  # or any other logged metric
)
```

**`get_best_model()` and other methods:**
```python
best_run = selector.get_best_model()
best_runs_by_tag = selector.get_best_models_by_tag("base_model_name")
```

**`ModelTransition()` Arguments:**
```python
ModelTransition(
    model_name="...",
    registry_uri="..."  # Defaults to "databricks"
)
```

**`transition_model()` call:**
```python
transitioner.transition_model(
    stage="Production",
    experiment_name="...",
    metric="accuracy",
    version=None  # If None, it picks the best run
)
```

---

## Summary
This custom MLflow module streamlines the entire lifecycle:
1. **Experiment Setup & Run Execution:** Organized by `ExperimentManager`, which handles training and evaluation runs.
2. **Model Comparison & Selection:** Assisted by `ModelSelector`, which identifies best-performing models based on metrics.
3. **Model Registry & Deployment:** With `ModelTransition`, models move from experiment runs to registered versions and into production stages in the MLflow Model Registry.

By following this documentation and using the provided classes and methods, non-technical stakeholders can efficiently track, evaluate, and deploy models into production, ensuring a smooth, end-to-end machine learning workflow.
```