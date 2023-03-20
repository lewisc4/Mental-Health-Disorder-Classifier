import numpy as np
import pandas as pd

from stellargraph import StellarGraph, IndexedArray
from sklearn.utils.class_weight import compute_class_weight


class RedditNetwork:

	def __init__(self, args, nodes, edges, features=None, ground_truth=None):
		self.args = args
		self.nodes = nodes
		self.edges = edges
		self.features = features
		self.ground_truth = ground_truth


	@property
	def nodes(self):
		return self._nodes
	

	@nodes.setter
	def nodes(self, nodes):
		col_names = {
			self.args.node_id_col: 'ids',
			self.args.edge_id_col: 'users',
			self.args.label_col: 'labels',
			self.args.label_name_col: 'label_names',
			self.args.data_col: 'text',
		}
		nodes.rename(columns=col_names, inplace=True)
		nodes.ids = pd.to_numeric(nodes.ids)
		self._nodes = nodes.set_index('ids', drop=False, verify_integrity=True)


	@property
	def edges(self):
		return self._edges


	@edges.setter
	def edges(self, edges):
		col_names = {
			self.args.src_col: 'src',
			self.args.tgt_col: 'tgt',
		}
		edges.rename(columns=col_names, inplace=True)
		valid = self.nodes.index
		edges = edges[edges.src.isin(valid) & edges.tgt.isin(valid)]
		self._edges = edges.set_index(['src', 'tgt'], drop=False)


	@property
	def features(self):
		return self._features
	

	@features.setter
	def features(self, features):
		if features is None:
			self._features = IndexedArray()
		else:
			self._features = IndexedArray(features, index=self.nodes.index)


	@property
	def ground_truth(self):
		return self._ground_truth


	@ground_truth.setter
	def ground_truth(self, ground_truth):
		if ground_truth is None:
			ground_truth = []
		self._ground_truth = self.nodes.users.isin(ground_truth)


	def to_stellargraph(self):
		return StellarGraph(
			nodes=self.features,
			edges=self.edges,
			source_column='src',
			target_column='tgt',
			edge_weight_column=None,
			node_type_default='Reddit comment',
			edge_type_default='Comment belongs to same user'
		)


	def get_label_id_map(self):
		label_id_map = self.nodes.set_index('labels')['label_names'].to_dict()
		return dict(sorted(label_id_map.items()))


	def to_dataset(self):
		test_df = self.nodes[self.ground_truth]
		train_df = self.nodes[~self.ground_truth]
		train_df, val_df = self.train_val_split(train_df)
		return {
			'train': train_df[['ids', 'labels']],
			'val': val_df[['ids', 'labels']],
			'test': test_df[['ids', 'labels']],
		}


	def train_val_split(self, original_df):
		label_groups = original_df.groupby(
			'labels',
			as_index=False,
			group_keys=False
		)
		percent_train = self.args.percent_train
		train_df = label_groups.apply(lambda x: x.sample(frac=percent_train))
		val_df = original_df.drop(train_df.index)
		return train_df, val_df


	def get_class_weights(self, dataset, split='train'):
		unique_classes = np.unique(dataset[split].labels)
		weights = compute_class_weight(
			class_weight='balanced',
			classes=unique_classes,
			y=dataset[split].labels
		)
		return dict(zip(unique_classes, weights))

