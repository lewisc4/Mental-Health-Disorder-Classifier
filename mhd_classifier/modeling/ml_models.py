import pandas as pd

from abc import ABC, abstractmethod
from sklearn import preprocessing
from stellargraph.mapper import FullBatchNodeGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GCN, GAT, GraphSAGE
from tensorflow.keras import layers, optimizers, losses, Model


class GraphMLModel(ABC):

	def __init__(self, graph, train_data, val_data, test_data, args, trained_model):
		self.graph = graph
		self.train_data = train_data
		self.val_data = val_data
		self.test_data = test_data
		self.args = args
		self.trained_model = trained_model

		self.init_generator()
		self.init_graph_model()
		self.init_targets()
		self.init_nodes()


	@abstractmethod
	def init_generator(self):
		self.generator = None
		self.shuffleable_generator = None


	@abstractmethod
	def init_graph_model(self):
		self.graph_model = None


	def init_targets(self):
		train_labels = self.train_data.labels.values
		val_labels = self.val_data.labels.values
		self.target_encoding = preprocessing.LabelBinarizer()
		self.train_targets = self.target_encoding.fit_transform(train_labels)
		self.val_targets = self.target_encoding.transform(val_labels)


	def init_nodes(self):
		if self.shuffleable_generator:
			self.train_nodes = self.generator.flow(
				self.train_data.index,
				self.train_targets,
				shuffle=self.args.shuffle_generator
			)
		else:
			self.train_nodes = self.generator.flow(
				self.train_data.index,
				self.train_targets
			)

		self.val_nodes = self.generator.flow(
			self.val_data.index,
			self.val_targets
		)
		self.test_nodes = self.generator.flow(
			self.test_data.index
		)


	def train(self, save=True, class_weight=None):
		x_in, x_out = self.graph_model.in_out_tensors()
		pred_layer = layers.Dense(
			units=self.train_targets.shape[1],
			activation=self.args.final_activation
		)(x_out)

		model = Model(inputs=x_in, outputs=pred_layer)
		model.compile(
			optimizer=optimizers.Adam(learning_rate=self.args.learning_rate),
			loss=losses.categorical_crossentropy,
			metrics=['acc']
		)
		_ = model.fit(
			self.train_nodes,
			epochs=self.args.num_train_epochs,
			validation_data=self.val_nodes,
			verbose=2,
			shuffle=self.args.shuffle_train_data,
		)
		if save:
			model.save(self.args.model_dir / self.args.model_location)

		self.trained_model = model
		return model


	def test(self):
		if self.trained_model is None:
			return

		test_ids = self.test_data.index
		test_labels = self.test_data.labels

		embeddings = self.trained_model.predict(self.test_nodes, verbose=1)
		embeddings = embeddings.squeeze()

		predicted_labels = self.target_encoding.inverse_transform(embeddings)
		predictions_df = pd.DataFrame(
			{
				'node_id': test_ids,
				'predicted': predicted_labels,
				'actual': test_labels
			}
		)
		return embeddings, predictions_df


class GraphSAGEModel(GraphMLModel):

	def __init__(self, graph, train_data, val_data, test_data, args, trained_model=None):
		# Special case, always shuffle train data (in generator) for GraphSAGE
		args.shuffle_generator = True
		super().__init__(graph, train_data, val_data, test_data, args, trained_model)


	def init_generator(self):
		self.generator = GraphSAGENodeGenerator(
			G=self.graph,
			batch_size=self.args.batch_size,
			num_samples=self.args.layer_num_samples
		)
		self.shuffleable_generator = True



	def init_graph_model(self):
		self.graph_model = GraphSAGE(
			layer_sizes=[self.args.layer_size] * self.args.num_layers,
			generator=self.generator,
			bias=self.args.use_bias,
			dropout=self.args.dropout
		)


class GCNModel(GraphMLModel):

	def __init__(self, graph, train_data, val_data, test_data, args, trained_model=None):
		super().__init__(graph, train_data, val_data, test_data, args, trained_model)


	def init_generator(self):
		self.generator = FullBatchNodeGenerator(G=self.graph, method='gcn')
		self.shuffleable_generator = False



	def init_graph_model(self):
		self.graph_model = GCN(
			layer_sizes=[self.args.layer_size] * self.args.num_layers,
			generator=self.generator,
			activations=[self.args.hidden_activation] * self.args.num_layers,
			dropout=self.args.dropout
		)


class GATModel(GraphMLModel):

	def __init__(self, graph, train_data, val_data, test_data, args, trained_model=None):
		super().__init__(graph, train_data, val_data, test_data, args, trained_model)


	def init_generator(self):
		self.generator = FullBatchNodeGenerator(G=self.graph, method='gat')
		self.shuffleable_generator = False



	def init_graph_model(self):
		self.graph_model = GAT(
			layer_sizes=[self.args.layer_size] * self.args.num_layers,
			generator=self.generator,
			activations=[self.args.hidden_activation] * self.args.num_layers,
			attn_heads=self.args.num_attn_heads,
			in_dropout=self.args.dropout,
			attn_dropout=self.args.attn_dropout,
			normalize=None
		)

