import re
import emoji
import numpy as np

from string import digits, punctuation
from unidecode import unidecode
from tqdm import tqdm
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer


class Preprocessor:

	def __init__(self):
		# Diagnosis patterns to remove words related to disorders before embedding
		self.bad_chars = digits + punctuation
		self.url_regex = re.compile(r'http\S+')
		self.stopwords = list(set(stopwords.words('english')))


	def clean(self, df, field='text', exclude_len=3):
		''' 
		Removes invalid chars, emojis, urls, converts non-ascii chars
		Takes a Pandas DF and one of its columns to clean, returns the cleaned DF
		'''
		# For each row in the Pandas column, clean the text
		to_clean = tqdm(df[field].to_numpy(), desc='Cleaning')
		df[field] = [self.clean_text(text) for text in to_clean]
		excluded = df[field].str.split().str.len() <= exclude_len
		return df[~excluded]


	def clean_text(self, text):
		# Remove new lines, double spaces etc
		single_spaces = ' '.join(text.split())
		# Remove urls, convert emojis to text, convert non-ascii to ascii
		no_urls = self.url_regex.sub('', single_spaces)
		no_emojis = emoji.demojize(no_urls)
		ascii_text = unidecode(no_emojis).lower()
		# Remove invalid chars from string
		valid = ascii_text.translate({ord(c): None for c in self.bad_chars})
		# Remove any extra whitespace left after removing invalid chars
		return ' '.join(valid.split())


	def embed(self, embedding_type, df, field='text'):
		''' Given a Pandas DF and one of its columns, embed that data '''
		embedding_model = SentenceTransformer(embedding_type)
		text = df[field].tolist()
		embeddings = embedding_model.encode(text, show_progress_bar=True)
		return np.array(embeddings)


	def remove_stopwords(self, df, field='text', custom_stopwords=None):
		''' Removes stop words from Pandas DF column'''
		if custom_stopwords is not None:
			self.stopwords += custom_stopwords
		to_search = tqdm(df[field].to_numpy(), desc='Removing Stopwords')
		df[field] = [self.remove_stopwords_from(text) for text in to_search]
		return df


	def remove_stopwords_from(self, text):
		removed = [word for word in text.split() if word not in self.stopwords and len(word) > 1]
		return ' '.join(removed)

