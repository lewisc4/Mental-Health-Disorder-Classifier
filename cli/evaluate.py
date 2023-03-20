from mhd_classifier.utils.cli_utils import parse_train_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.model_evaluations import ModelEvaluator


def main():
	# Parse script args
	args = parse_train_args()

	# File handler to use for reading and writing data
	f_manager = FileManager()	
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)

	nodes = f_manager.get('cleaned_network.csv')
	edges = f_manager.get('edge_list.csv')
	embeddings = f_manager.get('embeddings.npy')
	diagnosed_users = f_manager.get('diagnosed_users.txt')

	test_predictions = f_manager.get('test_predictions.csv')
	test_embeddings = f_manager.get('test_embeddings.npy')

	reddit_network = RedditNetwork(
		args=args,
		nodes=nodes,
		edges=edges,
		features=embeddings,
		ground_truth=diagnosed_users
	)
	label_id_map = reddit_network.get_label_id_map()

	model_evaluator = ModelEvaluator(
		predictions=test_predictions,
		embeddings=test_embeddings,
		pred_col='predicted',
		true_col='actual',
		label_id_map=label_id_map
	)

	metrics = model_evaluator.get_metrics()
	print(metrics)

	conf_mat = model_evaluator.get_confusion_matrix()
	conf_mat.show()

	embedding_visuals = model_evaluator.get_pred_visuals()
	embedding_visuals.show()


if __name__ == '__main__':
	main()

