import argparse
import sklearn

from pathlib import Path


# The CWD of the script importing this module
CWD = Path().resolve()

# The "root" data dir, containing all project resource files (by default)
DATA_DIR = CWD / 'data'
# Where the initial dataset (nodes/edges) is stored
DATASET_DIR = DATA_DIR / 'dataset'
# Locations of all diagnosis pattern-related files
PATTERN_DIR = DATA_DIR / 'keywords/patterns'
SYNONYM_DIR = DATA_DIR / 'keywords/synonyms'
# Directory where all processed data is written to (cleaned text, embeddings, etc.)
PROCESSED_DIR = DATA_DIR / 'processed'
# Directory for saving and loading models
MODEL_DIR = DATA_DIR / 'models'
# Directory where evaluation metric-related files are saved to
METRIC_DIR = DATA_DIR / 'evaluation_metrics'


def create_parser():
	'''
	This function creates a default, general argument parser to act as a 
	parent for scripts in this project that are run from the CLI.
	This function is used in preprocess.py, train.py, and evaluate.py.
	Default arguments have the meaning of being a reasonable default value.
	'''

	# General-purpose parser to use for different scripts
	# Not intended to be used as a standalone parser in this project
	parser = argparse.ArgumentParser(
		description='General-use parser for CLI scripts in this project',
		add_help=False # Don't add help, only because this is a parent parser
	)

	# File path arguments used throughout CLI scripts
	parser.add_argument(
		'--dataset_dir',
		type=Path,
		default=DATASET_DIR,
		help='Path to the directory where the dataset files are stored.',
	)
	parser.add_argument(
		'--pattern_dir',
		type=Path,
		default=PATTERN_DIR,
		help='Path to the directory where diagnosis pattern-related files are stored.',
	)
	parser.add_argument(
		'--synonym_dir',
		type=Path,
		default=SYNONYM_DIR,
		help='Path to the directory where files containing their respective synonyms are stored.',
	)
	parser.add_argument(
		'--processed_dir',
		type=Path,
		default=PROCESSED_DIR,
		help='Path to the directory where the pre-processed (i.e. cleaned) data is saved.',
	)
	parser.add_argument(
		'--model_dir',
		type=Path,
		default=MODEL_DIR,
		help='Path to the directory where trained models are located.',
	)
	parser.add_argument(
		'--metric_dir',
		type=Path,
		default=METRIC_DIR,
		help='Path to the directory where evaluation metric files are saved.'
	)

	# Dataset arguments (i.e. network file names and their column header names)
	parser.add_argument(
		'--node_id_col',
		type=str,
		default='node_ids',
		help='Column header corresponding to the network node id column.',
	)
	parser.add_argument(
		'--edge_id_col',
		type=str,
		default='users',
		help='Column header corresponding to the network edge id (i.e. what relates the src and tgt).',
	)
	parser.add_argument(
		'--label_col',
		type=str,
		default='subreddit_ids',
		help='Column header corresponding to the network label (class) ID column.',
	)
	parser.add_argument(
		'--label_name_col',
		type=str,
		default='subreddits',
		help='Column header corresponding to the network label (class) name column.',
	)
	parser.add_argument(
		'--data_col',
		type=str,
		default='comments',
		help='Column header corresponding to the network label (class) column.',
	)
	parser.add_argument(
		'--src_col',
		type=str,
		default='source_nodes',
		help='Column header corresponding to the source node column.',
	)
	parser.add_argument(
		'--tgt_col',
		type=str,
		default='target_nodes',
		help='Column header corresponding to the target node column.',
	)
	
	# Return parser to use as a parent, so we DON'T want to call parse_args()
	return parser


def parse_preprocessing_args(parents=[]):
	'''
	This function creates a preprocessing-related argument parser and parses a
	(preprocessing) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example, assuming:
	we have a parent parser with a dataset_dir argument:

	python3 preprocess.py --dataset_dir=dataset --embedding_model=average_word_embeddings_glove.840B.300d
	'''

	# Create parser based on the parent parser(s)
	# An empty list ([]) is equivalent to no parents
	parser = argparse.ArgumentParser(
		parents=parents,
		description='Pre-process a network/graph of reddit comment data'
	)

	# Data preprocessing-related parameters
	# See www.sbert.net/docs/pretrained_models.html for pre-trained examples
	parser.add_argument(
		'--embedding_model',
		type=str,
		default='average_word_embeddings_glove.6B.300d',
		help='The (word) embedding model to use for feature generation.'
	)

	args = parser.parse_args()
	return args


def parse_train_args(parents=[]):
	'''
	This function creates a training-related argument parser and parses a
	(training) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example, assuming:
	we have a parent parser with a processed_dir argument:

	python3 train.py --processed_dir=dataset --learning_rate=2e-3
	'''

	# Create parser based on the parent parser(s)
	# An empty list ([]) is equivalent to no parents
	parser = argparse.ArgumentParser(
		parents=parents,
		description='Train a model on a network/graph of reddit comment data'
	)

	parser.add_argument(
		'--train_val_size',
		type=int,
		default=None,
		help='Combined size (# samples) of the training and validation set.',
	)
	parser.add_argument(
		'--test_size',
		type=int,
		default=None, # None implies a separate test set will be used
		help='Size (# samples) of the test set.',
	)
	parser.add_argument(
		'--percent_train',
		type=float,
		default=0.8,
		help='Percentage of the data to use for training (train_val_size * percent_train).',
	)
	parser.add_argument(
		'--shuffle_generator',
		type=bool,
		default=True,
		help='Whether to shuffle training data.',
	)
	parser.add_argument(
		'--shuffle_train_data',
		type=bool,
		default=True,
		help='Whether to shuffle training data.',
	)

	# Training arguments
	parser.add_argument(
		'--model_type',
		type=str,
		default='GraphSAGE',
		choices=['GraphSAGE', 'GCN', 'GAT'],
		help='The type of graph model to train.'
	)
	parser.add_argument(
		'--model_location',
		type=str,
		default='trained_model',
		help='The model name to use for saving and loading a model.'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=256,
		help='Batch size (per device) for the training dataloader.',
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=5e-4,
		help='Initial learning rate (after the potential warmup period) to use.',
	)
	parser.add_argument(
		'--num_train_epochs',
		type=int,
		default=5,
		help='Total number of training epochs to perform.',
	)
	parser.add_argument(
		'--steps_per_epoch',
		type=int,
		default=540,
		help='Number of (update) steps to take during each training epoch.',
	)
	parser.add_argument(
		'--eval_every_steps',
		type=int,
		default=40,
		help='Perform evaluation every n network updates.',
	)
	parser.add_argument(
		'--dropout',
		type=float,
		default=0.5,
		help='Dropout rate to use during training.',
	)
	parser.add_argument(
		'--num_layers',
		type=int,
		default=2,
		help='Number of hidden layers to use during training.',
	)
	parser.add_argument(
		'--layer_size',
		type=int,
		default=16,
		help='Size of hidden layers to use during training.',
	)
	parser.add_argument(
		'--hidden_activation',
		type=str,
		default='relu',
		help='Activation function to use for hidden layers.',
	)
	parser.add_argument(
		'--final_activation',
		type=str,
		default='softmax',
		help='Final activation function to use at final layer.',
	)
	parser.add_argument(
		'--num_attn_heads',
		type=int,
		default=8,
		help='Number of attention heads to use during training.',
	)
	parser.add_argument(
		'--attn_dropout',
		type=float,
		default=0.5,
		help='Dropout rate to use for attention during training.',
	)
	parser.add_argument(
		'--use_bias',
		type=bool,
		default=True,
		help='Whether to use bias during training or not.',
	)
	parser.add_argument(
		'--layer_num_samples',
		'--list',
		type=lambda s: [int(num_samples) for num_samples in s.split(',')],
		default=[30, 10],
		help='List of comma separated number of samples to use at each layer.'
	)

	args = parser.parse_args()
	return args


def parse_eval_args(parents=[]):
	'''
	This function creates a model evaluation-related argument parser and
	parses a (model evaluation) script's input arguments.

	Default arguments have the meaning of being a reasonable default value.
	To change the parameters, pass them to the script. For example, assuming:
	we have a parent parser with a metric_dir argument:

	python3 evaluate.py --metric_dir=evaluation_metrics --metrics=scores,confusion_matrix
	'''

	# Create parser based on the parent parser(s)
	# An empty list ([]) is equivalent to no parents
	parser = argparse.ArgumentParser(
		parents=parents,
		description='Generate a evaluation metrics using model predictions.'
	)

	# Evaluation-related parameters
	parser.add_argument(
		'--metrics',
		'--list',
		type=lambda m: [str(metric) for metric in m.split(',')],
		default=['scores', 'confusion_matrix', 'embedding_visualization'],
		help='List of metrics/figures to generate. Defaults to all metrics.',
	)

	parser.add_argument(
		'--pred_col',
		type=str,
		default='predicted',
		help='Column header to use for predicted label IDs in test_predictions.csv.',
	)

	parser.add_argument(
		'--true_col',
		type=str,
		default='actual',
		help='Column header to use for true label IDs in test_predictions.csv.',
	)

	parser.add_argument(
		'--save_prefix',
		type=str,
		default='',
		help='Prefix to add to each metric file that is saved.',
	)
	parser.add_argument(
		'--title_prefix',
		type=str,
		default='',
		help='Prefix to use for figure titles (for metrics that use a title).'
	)

	args = parser.parse_args()
	return args

