from mhd_classifier.utils.cli_utils import create_parser, parse_eval_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.model_evaluations import ModelEvaluator


# File name constants, files required BEFORE evaluation
CLEANED_NETWORK_F = 'cleaned_network.csv'
EDGE_LIST_F = 'edge_list.csv'
EMBEDDINGS_F = 'embeddings.npy'
DIAGNOSED_USERS_F = 'diagnosed_users.txt'
TEST_PREDICTIONS_F = 'test_predictions.csv'
TEST_EMBEDDINGS_F = 'test_embeddings.npy'


def main():
	# Parse script args
	script_parser = create_parser()
	args = parse_eval_args(parents=[script_parser])

	# File handler to use for reading and writing data
	f_manager = FileManager()	
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)

	# Update the root directory to use for file I/O
	f_manager.root = args.metric_dir

	# Get the data required to build our Reddit network
	nodes = f_manager.get(CLEANED_NETWORK_F)
	edges = f_manager.get(EDGE_LIST_F)
	embeddings = f_manager.get(EMBEDDINGS_F)
	diagnosed_users = f_manager.get(DIAGNOSED_USERS_F)

	# Get the test predictions/embeddings used for evaluation
	test_predictions = f_manager.get(TEST_PREDICTIONS_F)
	test_embeddings = f_manager.get(TEST_EMBEDDINGS_F)

	# Build our Reddit network
	reddit_network = RedditNetwork(
		args=args,
		nodes=nodes,
		edges=edges,
		features=embeddings,
		ground_truth=diagnosed_users
	)
	# Get the network's label-ID map, useful for plotting class names
	label_id_map = reddit_network.get_label_id_map()

	# Create our ModelEvaluator class to get metrics using the predictions
	model_evaluator = ModelEvaluator(
		predictions=test_predictions,
		embeddings=test_embeddings,
		label_id_map=label_id_map,
		pred_col=args.pred_col,
		true_col=args.true_col,
		title_prefix=args.title_prefix
	)

	# Get and save each metric from the user-provided metric names
	for metric_name in args.metrics:
		metric = model_evaluator.get_metric(metric_name)
		# Filename prefix used to create unique/descriptive metric filenames
		f_manager.write(args.save_prefix + metric_name, metric)


if __name__ == '__main__':
	main()

