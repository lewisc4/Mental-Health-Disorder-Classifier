# Mental Health Disorder Classifier Using Graph Machine Learning

## Project Overview
In this project, three graph-based machine learning models were trained to classify five mental health disorders (anxiety, depression, ADHD, eating disorders, and OCD) using user-comment data from [Reddit](https://www.reddit.com/), which was scraped using the [Pushshift API](https://github.com/pushshift/api) and [Google's BigQuery](https://cloud.google.com/bigquery). Using this data, a large scale [network dataset](https://github.com/lewisc4/Mental-Health-Disorder-Classifier/blob/main/code/cli/data.zip) (i.e. a graph) was generated using comments from subreddits related to the five mental health disorders, as well as one control subreddit ([r/gaming](https://www.reddit.com/r/gaming/)). Each comment represents a node in the network and nodes that share the same author are linked to one another (that is, an edge is formed between them). Three graph-based machine learning frameworks were used to generate low-dimensional representations for nodes in the network: [GraphSAGE](https://snap.stanford.edu/graphsage/), [GCN](https://tkipf.github.io/graph-convolutional-networks/), and [GAT](https://petar-v.com/GAT/).

## Setting Up The Environment
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the `code\` directory, where the `setup.py` file is located.
3. Install the `mhd_classifier` module and all dependencies by running the following command from the CLI: `pip install -e .` (required python modules are in `requirements.txt`).

### Downloading The Dataset
The dataset is stored in `data.zip`, located in this GitHub repository. Once this project has been cloned or downloaded to your local computer, unzip `data.zip`. By default, this project assumes **and requires** that it will be unzipped in its original directory (`code/cli`), where the CLI scripts are located. Once it has been unzipped, `data.zip` can be deleted.

<!--
After unzipping `data.zip`, you will find it has the following initial structure:
Please see `README.md` in the `code/cli` directory for more information on the dataset.

## Training A Model
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

### CLI Training Commands
**Make sure you have followed the `Setting Up The Environment` section before running these commands**

The below commands can be run from the `cli` directory. By default, the model is saved to the `code/cli/outputs/` directory. If the provided `output_dir` does not exist, it will automatically be created.

**To train a model with the parameters that achieved the best accuracy:**

`python3 train.py --test_for_val --test_type='public_test' --pretrained_model_name='microsoft/resnet-50' --batch_size=64 --learning_rate=1e-3 --lr_scheduler_type='linear' --weight_decay=0.0 --num_train_epochs=30 --eval_every_steps=90 --logging_steps=90 --checkpoint_every_steps=10000 --seed=42`

**To perform wandb sweeps using the `sweep.yaml` configuration file (make sure you have set a wandb project using the wandb_project argument):**

1. `wandb sweep --project emotion_detection sweep.yaml`
2. `wandb agent wandb_username/emotion_detection/sweep_id`
-->
