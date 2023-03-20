import json
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score


class ModelEvaluator:

	def __init__(self, predictions, embeddings, pred_col, true_col, label_id_map):
		self.predictions = predictions
		self.embeddings = embeddings
		
		self.pred_col = pred_col
		self.true_col = true_col
		
		self.y_pred = self.predictions[self.pred_col]
		self.y_true = self.predictions[self.true_col]
		
		self.label_id_map = label_id_map

		self.labels = list(self.label_id_map.values())
		self.pred_labels = self.y_pred.map(self.label_id_map)
		self.true_labels = self.y_true.map(self.label_id_map)


	def get_metrics(self):
		metric_keys = ['Precision', 'Recall', 'F1_Score']
		metrics = score(self.true_labels, self.pred_labels, labels=self.labels)

		label_metrics = [dict(zip(self.labels, metric)) for metric in metrics]
		grouped_metrics = dict(zip(metric_keys, label_metrics))
		return json.dumps(grouped_metrics, sort_keys=True, indent=4)


	def get_confusion_matrix(self, title='Confusion Matrix'):
		fig, ax = plt.subplots(figsize=(10, 10))
		plt.title(title)

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
		return plt


	def get_pred_visuals(self, n_components=2, alpha=0.7, title='Pred Visuals'):
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
		# plt_labels has '$\\mathdefault{0}$', '$\\mathdefault{1}$', etc. instead of ints
		label_ids = [int(''.join(i for i in x if i.isdigit())) for x in plt_labels]
		embedding_labels = [self.label_id_map[lbl_id] for lbl_id in label_ids]

		plt.legend(handles=handles, labels=embedding_labels)
		plt.title(title)
		return plt

