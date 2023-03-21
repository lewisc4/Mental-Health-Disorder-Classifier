# Mental Health Disorder Classification Using Graph-Based Machine Learning

## Project Overview
In this project, three graph-based machine learning models were trained to classify five mental health disorders (anxiety, depression, ADHD, eating disorders, and OCD) using user-comment data from [Reddit](https://www.reddit.com/), which was scraped using the [Pushshift API](https://github.com/pushshift/api) and [Google's BigQuery](https://cloud.google.com/bigquery). Using this data, a large scale [network dataset](/cli/data.zip) (i.e. a graph) was generated using comments from subreddits related to the five mental health disorders, as well as one control subreddit ([r/gaming](https://www.reddit.com/r/gaming/)). Each comment represents a node in the network and nodes that share the same author are linked to one another (that is, an edge is formed between them). Three graph-based machine learning frameworks were used to generate low-dimensional representations for nodes in the network: [GraphSAGE](https://snap.stanford.edu/graphsage/), [GCN](https://tkipf.github.io/graph-convolutional-networks/), and [GAT](https://petar-v.com/GAT/).

## Environment Setup
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the [root directory](https://github.com/lewisc4/Mental-Health-Disorder-Classifier), where the [`setup.py`](/setup.py) file is located.
3. Install the [`mhd_classifier`](/mhd_classifier) module and all dependencies by running the following command: `pip install -e .` (required python modules are in [`requirements.txt`](/requirements.txt)).

### Downloading The Dataset
The dataset is stored in [`data.zip`](/cli/data.zip), located in this GitHub repository. Once this project has been cloned or downloaded to your local computer, unzip [`data.zip`](/cli/data.zip). This project assumes **and requires** that it will be unzipped in its original directory ([`cli/`](/cli)), where all scripts run from the CLI located. Once unzipped, [`data.zip`](/cli/data.zip) can be deleted.

Unzipping [`data.zip`](/cli/data.zip) will reveal the following file structure:

```bash
data
├── dataset # Contains Reddit network dataset files
│   ├── edge_list.csv # Each row has a pair of (source, target) comment IDs
│   └── network.csv # Each row has (comment ID, username, subreddit name, subreddit ID, comment text)
│ 
└── keywords # Contains pattern-building data for identifying self-diagnosed users
    ├── patterns # Contains files to build self-diagnosis pattern bases and variations
    │   ├── common_prefixes.txt # Common words (e.g. my, the, a) to create sentence variations
    │   └── diagnosis_patterns.txt # Generic self-diagnosis patterns (e.g. "I was diagnosed with __")
    │
    └── synonyms # Contains files with diagnosis-related synonyms, to inject in base patterns
        ├── adhd.txt # Synonyms for ADHD (ADD, attention deficit hyperactivity, etc.)
        ├── anxiety.txt # Synonyms for anxiety (GAD, generalized anxiety disorder, etc.)
        ├── depression.txt # Synonyms for depression (MDD, major depressive disorder, etc.)
        ├── doctor.txt # Synonyms for doctor (doc, psychiatrist, etc.)
        ├── eating_disorders.txt # Synonyms for eating disorders (anorexia, bulimia, etc.)
        └── ocd.txt # Synonyms for OCD (obsessive compulsive disorder, anankastic neurosis, etc.)
```

<!--
## Training
### Hyperparameters
The available hyperparameters for fine-tuning the ResNet model can be found in the `emotion_detection/utils.py` file. By default, a large majority of the hyperparameters are inherited from the ResNet model's original parameters. The default model is `microsoft/resnet-18`. Useful parameters to change/test with are:

* `data_dir` <- Parent folder of the dataset (`dataset` by default). See the `Downloading The Dataset` section for more.
* `output_dir` <- Where to save the model to (defaults to `code/cli/outputs/`)
* `test_for_val` <- Whether to use the test set for validation or not. If not, a subset of the training data is used.
* `test_type` <- Uses either public test set (`public_test.csv`) or the private test set (`private_test.csv`) if the test set is used for validation.
* `percent_train` <- What percentage of the training dataset should be used for training, if a subset is used as a validation set.
* `learning_rate` <- The external learning rate
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay
* `eval_every_steps` <- How often to evaluate the model (compute eval accuracy)
* `debug` <- Whether to run in debug mode (uses small number of examples) or not
* `num_train_epochs` <- Number of training epochs to use
* `wandb_project` <- The weights and biases project to use (not required)
* `use_wandb` <- Whether to log to weights and biases or not (do not use unless you have a project set via `wandb_project`)

### Example Usage

**To train a model with the parameters that achieved the best accuracy:**
- `python3 train.py --test_for_val --test_type='public_test' --pretrained_model_name='microsoft/resnet-50' --batch_size=64 --learning_rate=1e-3 --lr_scheduler_type='linear' --weight_decay=0.0 --num_train_epochs=30 --eval_every_steps=90 --logging_steps=90 --checkpoint_every_steps=10000 --seed=42`
-->
