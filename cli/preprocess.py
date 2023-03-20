from mhd_classifier.utils.cli_utils import parse_preprocessing_args, parse_train_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.utils.preprocessing_utils import Preprocessor
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.diagnosis_helper import Diagnoser


def main():
	# Parse script args
	args = parse_preprocessing_args()
	args_train = parse_train_args()
	# File handler to use for reading and writing preprocessed data
	f_manager = FileManager()
	# The preprocesser helper class we will use
	preprocessor = Preprocessor()
	
	# Read files from necessary directories
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)
	f_manager.read(args.pattern_dir)
	f_manager.read(args.synonym_dir)

	# Get necessary files
	network = f_manager.get('network.csv')
	edge_list = f_manager.get('edge_list.csv')
	diagnosis_patterns = f_manager.get('diagnosis_patterns.txt')
	common_prefixes = f_manager.get('common_prefixes.txt')
	synonyms = f_manager.get_dir('synonyms')

	# Build network and clean data
	reddit_network = RedditNetwork(
		args=args_train,
		nodes=network,
		edges=edge_list
	)
	cleaned = preprocessor.clean(df=reddit_network.nodes, field='text')
	f_manager.write(args.processed_dir, 'cleaned_network.csv', cleaned)

	# Use cleaned data to get diagnosed users
	diagnoser = Diagnoser(
		disorder_list=args.disorder_names,
		patterns=diagnosis_patterns,
		synonym_map=synonyms,
		common_prefixes=common_prefixes
	)
	diagnosed_users = diagnoser.get_diagnoses(cleaned)
	f_manager.write(args.processed_dir, 'diagnosed_users.txt', diagnosed_users)

	# Get custom stopwords and remove them from cleaned text
	# After removing stopwords, embed the text to use as features
	disorder_syns = [synonyms[d] for d in args.disorder_names]
	all_synonyms = [syn for syn_list in disorder_syns for syn in syn_list]
	no_stopwords = preprocessor.remove_stopwords(
		df=cleaned,
		field='text',
		custom_stopwords=all_synonyms
	)
	embeddings = preprocessor.embed(
		df=no_stopwords,
		field='text',
		embedding_type=args.embedding_model
	)
	f_manager.write(args.processed_dir, 'embeddings.npy', embeddings)


if __name__ == '__main__':
	main()

