from mhd_classifier.utils.cli_utils import create_parser, parse_preprocessing_args
from mhd_classifier.utils.file_utils import FileManager
from mhd_classifier.utils.preprocessing_utils import Preprocessor
from mhd_classifier.modeling.reddit_network import RedditNetwork
from mhd_classifier.modeling.diagnosis_helper import Diagnoser


# File name constants, files required BEFORE preprocessing
NETWORK_F = 'network.csv'
EDGE_LIST_F = 'edge_list.csv'
DIAGNOSIS_PATTERN_F = 'diagnosis_patterns.txt'
COMMON_PREFIX_F = 'common_prefixes.csv'
SYNONYMS_DIR = 'synonyms'

# File name constants, files written AFTER preprocessing
CLEANED_NETWORK_F = 'cleaned_network.csv'
DIAGNOSED_USERS_F = 'diagnosed_users.txt'
EMBEDDINGS_F = 'embeddings.npy'

# Mental health disorder names (i.e. labels) used in this project
DISORDER_NAMES = ['anxiety', 'depression', 'adhd', 'ocd', 'eating_disorders']


def main():
	# Parse script args
	script_parser = create_parser()
	args = parse_preprocessing_args(parents=[script_parser])

	# File handler to use for reading and writing preprocessed data
	f_manager = FileManager()
	# The preprocesser helper class we will use
	preprocessor = Preprocessor()
	
	# Read files from necessary directories
	f_manager.read(args.dataset_dir)
	f_manager.read(args.processed_dir)
	f_manager.read(args.pattern_dir)
	f_manager.read(args.synonym_dir)

	# Update the root directory to use for file I/O
	f_manager.root = args.processed_dir

	# Get necessary files that we read from the directories above
	network = f_manager.get(NETWORK_F)
	edge_list = f_manager.get(EDGE_LIST_F)
	diagnosis_patterns = f_manager.get(DIAGNOSIS_PATTERN_F)
	common_prefixes = f_manager.get(COMMON_PREFIX_F)
	synonyms = f_manager.get_dir(SYNONYMS_DIR)

	# Build our Reddit network using the original data
	reddit_network = RedditNetwork(
		args=args,
		nodes=network,
		edges=edge_list
	)
	# Clean each node's text (i.e. Reddit comment) and save the cleaned data
	cleaned = preprocessor.clean(df=reddit_network.nodes)
	f_manager.write(CLEANED_NETWORK_F, cleaned)

	# Create a Diagnoser to identify Reddit comments containing self-diagnoses
	diagnoser = Diagnoser(
		disorder_list=DISORDER_NAMES,
		patterns=diagnosis_patterns,
		synonym_map=synonyms,
		common_prefixes=common_prefixes
	)
	# Search the cleaned Reddit comments for self-reported diagnoses
	diagnosed_users = diagnoser.get_diagnoses(cleaned)
	f_manager.write(DIAGNOSED_USERS_F, diagnosed_users)

	# Get the synonyms for each disorder
	disorder_syns = [synonyms[d] for d in DISORDER_NAMES]
	all_synonyms = [syn for syn_list in disorder_syns for syn in syn_list]

	# Remove stopwords from the cleaned data to generate feature embeddings
	# We consider the disorder synonyms to be stopwords and also remove them,
	# because we don't want to "cheat" by using them in our feature embeddings
	no_stopwords = preprocessor.remove_stopwords(
		df=cleaned,
		custom_stopwords=all_synonyms
	)
	# Get embeddings using the cleaned data (with no stopwords) and save them
	embeddings = preprocessor.embed(
		embedding_type=args.embedding_model,
		df=no_stopwords
	)
	f_manager.write(EMBEDDINGS_F, embeddings)


if __name__ == '__main__':
	main()

