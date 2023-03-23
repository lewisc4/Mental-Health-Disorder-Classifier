import json
import sklearn
import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score


# Score names to use/return from  the precision_recall_fscore_support function
SCORE_NAMES = ['Precision', 'Recall', 'F1_Score']


class ModelEvaluator:
	''' Class to calculate evaluation metrics using model predictions. '''

	def __init__(
		self,
		predictions,
		embeddings,
		label_id_map,
		pred_col='predicted',
		true_col='actual',
		title_prefix=''
	):
		# Model prediction DF and prediction embeddings
		self.predictions = predictions
		self.embeddings = embeddings
		# Dict to map label IDs (keys) to their corresponding label name
		self.label_id_map = label_id_map
		# DF columns for predicted and true (i.e. actual) label IDs
		self.pred_col = pred_col
		self.true_col = true_col
		# Prefix to use in figure titles (useful for differentiating plots)
		self.title_prefix = title_prefix.strip() + ' '
		# Predicted/true (i.e. actual) label IDs
		self.y_pred = self.predictions[self.pred_col]
		self.y_true = self.predictions[self.true_col]
		# Unique label names and all predicted/true (i.e. actual) label names
		self.labels = list(self.label_id_map.values())
		self.pred_labels = self.y_pred.map(self.label_id_map)
		self.true_labels = self.y_true.map(self.label_id_map)
		# Maps a metric name to its corresponding function
		self.metric_function_map = {
			'scores': self.get_scores,
			'confusion_matrix': self.get_confusion_matrix,
			'embedding_visualization': self.get_pred_visuals,
		}


	def get_metric(self, metric_name):
		''' Gets a metric based on its name. '''
		return self.metric_function_map.get(metric_name, lambda: None)()


	def get_scores(self):
		''' Get per-class precision, recall, and F1-scores from predictions. '''
		scores = score(self.true_labels, self.pred_labels, labels=self.labels)

		label_scores = [dict(zip(self.labels, metric)) for metric in scores]
		grouped_scores = dict(zip(SCORE_NAMES, label_scores))
		return json.dumps(grouped_scores, sort_keys=True, indent=4)


	def get_confusion_matrix(self):
		''' Gets a confusion matrix (matplotlib figure) from predictions. '''
		fig, ax = plt.subplots(figsize=(10, 10))
		plt.title(self.title_prefix + 'Confusion Matrix')
		conf_mat = confusion_matrix(
			self.true_labels,
			self.pred_labels,
			labels=self.labels,
			normalize='pred'
		)
		disp = ConfusionMatrixDisplay(
			confusion_matrix=conf_mat,
			display_labels=self.labels
		)
		disp.plot(ax=ax, cmap=plt.cm.Blues)
		# Return the current pyplot Figure
		return plt.gcf()


	def get_pred_visuals(self, n_components=2, alpha=0.7):
		''' Gets node embedding visualizations (matplotlib figure) from embeddings. '''
		# Project/transform the embeddings into 2-Dimensional space
		transform = TSNE
		trans = transform(n_components=n_components)
		embeddings_2d = trans.fit_transform(self.embeddings)
		
		# Plot 2-D embeddings for each class
		plt.figure(figsize=(7, 7))
		plt.axes().set(aspect='equal')
		scatter = plt.scatter(
			embeddings_2d[:, 0],
			embeddings_2d[:, 1],
			c=self.y_true, # Node colors for each class
			cmap='jet',
			alpha=alpha,
		)
		handles, plt_labels = scatter.legend_elements()
		label_ids = [int(''.join(i for i in x if i.isdigit())) for x in plt_labels]
		embedding_labels = [self.label_id_map[lbl_id] for lbl_id in label_ids]

		plt.legend(handles=handles, labels=embedding_labels)
		plt.title(self.title_prefix + 'Node Embedding Visualizations')
		# Return the current pyplot Figure
		return plt.gcf()

