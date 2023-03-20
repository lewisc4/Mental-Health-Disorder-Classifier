import argparse

from pathlib import Path


CWD = Path().resolve()
DATASET_DIR = Path(CWD, 'data/dataset')
PATTERN_DIR = Path(CWD, 'data/keywords/patterns')
SYNONYM_DIR = Path(CWD, 'data/keywords/synonyms')
MATCH_DIR = Path(CWD, 'data/keywords/matches')
PROCESSED_DIR = Path(CWD, 'data/processed')
MODEL_DIR = Path(CWD, 'data/models')
EVALUATION_RESULTS_DIR = Path(CWD, 'data/evaluation_results')
AVAILABLE_DISORDERS = ['anxiety', 'depression', 'adhd', 'ocd', 'eating_disorders']


def parse_preprocessing_args():
	'''
	This function creates argument parser and parses a script's input arguments.
	This is the most common way to define input arguments in python. It is used
	by preprocess.py.

	To change the parameters, pass them to the script, for example:

	python cli/preprocess.py \
		--dataset_dir output_dir \
		--embedding_model average_word_embeddings_glove.840B.300d
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	'''
	parser = argparse.ArgumentParser(description='Pre-process a network/graph of reddit comment data')

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
		'--match_dir',
		type=Path,
		default=MATCH_DIR,
		help='Path to the directory where files containing their respective diagnosis matches are stored.',
	)
	parser.add_argument(
		'--processed_dir',
		type=Path,
		default=PROCESSED_DIR,
		help='Path to the directory where the pre-processed (i.e. cleaned) data is stored.',
	)
	parser.add_argument(
		'--model_dir',
		type=Path,
		default=MODEL_DIR,
		help='Path to the directory where trained models are saved/stored.',
	)
	parser.add_argument(
		'--evaluation_results_dir',
		type=Path,
		default=EVALUATION_RESULTS_DIR,
		help='Path to the directory where model evaluation files (i.e. plots/figures) are stored.',
	)
	# See www.sbert.net/docs/pretrained_models.html for pre-trained examples
	parser.add_argument(
		'--embedding_model',
		type=str,
		default='average_word_embeddings_glove.6B.300d',
		help='The (word) embedding model to use for feature generation.'
	)
	parser.add_argument(
		'--disorder_names',
		'--list',
		nargs='+',
		type=lambda s: [str(disorder) for disorder in s.split(',')],
		default=AVAILABLE_DISORDERS,
		help='List of comma separated disorders to consider during preprocessing.'
	)

	args = parser.parse_args()
	return args


def parse_train_args():
	'''
	This function creates argument parser and parses a script's input arguments.
	This is the most common way to define input arguments in python. It is used
	by train.py.

	To change the parameters, pass them to the script, for example:

	python cli/train.py \
		--dataset_dir output_dir \
		--learning_rate 2e-3
	
	Default arguments have the meaning of being a reasonable default value, not of the last arguments used.
	'''
	parser = argparse.ArgumentParser(description='Train a model on a network/graph of reddit comment data')

	# Required arguments
	parser.add_argument(
		'--dataset_dir',
		type=Path,
		default=DATASET_DIR,
		help='Path to the directory where the dataset files are stored.',
	)
	parser.add_argument(
		'--processed_dir',
		type=Path,
		default=PROCESSED_DIR,
		help='Path to the directory where the pre-processed (i.e. cleaned) data is stored.',
	)
	parser.add_argument(
		'--model_dir',
		type=Path,
		default=MODEL_DIR,
		help='Path to the directory where trained models are saved/stored.',
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
		default=None,
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
	parser.add_argument(
		'--debug',
		default=False,
		action='store_true',
		help='Whether to use a small subset of the dataset for debugging.',
	)

	# Dataset arguments
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
		help='Column header corresponding to the network edge id (i.e. what groups/relates src and tgt).',
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
		help='The name of the file to save/load the model to/from.'
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
		nargs='+',
		type=lambda s: [int(num_samples) for num_samples in s.split(',')],
		default=[30, 10],
		help='List of comma separated number of samples to use at each layer.'
	)

	args = parser.parse_args()
	return args

