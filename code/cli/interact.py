from tensorflow.keras.models import load_model
from mhd_classifier.utils.cli_utils import parse_train_args
from mhd_classifier.utils.file_utils import FileManager


def main():
	# Parse script args
	args = parse_train_args()
	f_manager = FileManager()

	# Load a trained model
	loaded_model = load_model(args.model_dir / args.model_location)


if __name__ == '__main__':
	main()

