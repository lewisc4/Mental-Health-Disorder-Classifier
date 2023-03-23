from mhd_classifier.utils.cli_utils import create_parser, parse_train_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.utils.preprocessing_utils import Preprocessor
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.ml_models import GraphSAGEModel, GCNModel, GATModel


# File name constants, files required BEFORE training
CLEANED_NETWORK_F = 'cleaned_network.csv'
EDGE_LIST_F = 'edge_list.csv'
EMBEDDINGS_F = 'embeddings.npy'
DIAGNOSED_USERS_F = 'diagnosed_users.txt'

# File name constants, files written AFTER training
TEST_EMBEDDINGS_F = 'test_embeddings.npy'
TEST_PREDICTIONS_F = 'test_predictions.csv'

# Maps model_type key to its model class
MODEL_TYPE_MAP = {
	'GraphSAGE': GraphSAGEModel,
	'GCN': GCNModel,
	'GAT': GATModel,
}


def main():
	# Parse script args
	script_parser = create_parser()
	args = parse_train_args(parents=[script_parser])

	# File handler to use for reading and writing data
	f_manager = FileManager()
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)

	# Update the root directory to use for file I/O
	f_manager.root = args.processed_dir

	# Get necessary data to construct our network
	nodes = f_manager.get(CLEANED_NETWORK_F)
	edges = f_manager.get(EDGE_LIST_F)
	embeddings = f_manager.get(EMBEDDINGS_F)
	diagnosed_users = f_manager.get(DIAGNOSED_USERS_F)
	# Build our Reddit network
	reddit_network = RedditNetwork(
		args=args,
		nodes=nodes,
		edges=edges,
		features=embeddings,
		ground_truth=diagnosed_users
	)

	# Get StellarGraph graph and to a dataset representation of our network
	graph = reddit_network.to_stellargraph()
	dataset = reddit_network.to_dataset()
	# Example on getting class weights from on dataset (not actually used)
	class_weights = reddit_network.get_class_weights(dataset)

	# Instantiate a graph model using the provided model type and arguments
	model = MODEL_TYPE_MAP.get(args.model_type, GCNModel)(graph, dataset, args)
	# Train our model and save it
	trained_model = model.train()
	trained_model.save(args.model_dir / args.model_location)

	test_embeddings, predictions_df = model.test(model_to_use=trained_model)
	f_manager.write(TEST_EMBEDDINGS_F, test_embeddings)
	f_manager.write(TEST_PREDICTIONS_F, predictions_df)


if __name__ == '__main__':
	main()
	
	