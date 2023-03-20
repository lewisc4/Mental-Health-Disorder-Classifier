from mhd_classifier.utils.cli_utils import parse_train_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.utils.preprocessing_utils import Preprocessor
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.ml_models import GraphSAGEModel, GCNModel, GATModel


def instantiate_model(graph, train_data, val_data, test_data, args):
	if args.model_type == 'GraphSAGE':
		return GraphSAGEModel(
			graph=graph,
			train_data=train_data,
			val_data=val_data,
			test_data=test_data,
			args=args
		)
	elif args.model_type == 'GCN':
		return GCNModel(
			graph=graph,
			train_data=train_data,
			val_data=val_data,
			test_data=test_data,
			args=args
		)
	else:
		return GATModel(
			graph=graph,
			train_data=train_data,
			val_data=val_data,
			test_data=test_data,
			args=args
		)


def main():
	# Parse script args
	args = parse_train_args()
	# File handler to use for reading and writing data
	f_manager = FileManager()
	# The preprocesser helper class we will use
	preprocessor = Preprocessor()

	f_manager = FileManager()
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)

	nodes = f_manager.get('cleaned_network.csv')
	edges = f_manager.get('edge_list.csv')
	embeddings = f_manager.get('embeddings.npy')
	diagnosed_users = f_manager.get('diagnosed_users.txt')

	reddit_network = RedditNetwork(
		args=args,
		nodes=nodes,
		edges=edges,
		features=embeddings,
		ground_truth=diagnosed_users
	)

	graph = reddit_network.to_stellargraph()
	dataset = reddit_network.to_dataset()
	class_weights = reddit_network.get_class_weights(dataset)

	args.learning_rate = 0.02
	args.num_train_epochs = 60
	args.batch_size = 300
	args.layer_size = 32
	args.model_type = 'GCN'

	model = instantiate_model(
		graph=graph,
		train_data=dataset['train'],
		val_data=dataset['val'],
		test_data=dataset['test'],
		args=args
	)
	model.train(save=True, class_weight=class_weights)

	test_embeddings, predictions_df = model.test()
	f_manager.write(args.processed_dir, 'test_embeddings.npy', test_embeddings)
	f_manager.write(args.processed_dir, 'test_predictions.csv', predictions_df)


if __name__ == '__main__':
	main()
	
	