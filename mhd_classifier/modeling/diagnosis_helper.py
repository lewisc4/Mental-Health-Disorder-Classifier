import re

from tqdm import tqdm


class PatternBuilder:

	def __init__(self, to_replace=None, base_patterns=None):
		self.to_replace = to_replace
		self.base_patterns = base_patterns


	@property
	def to_replace(self):
		return self._to_replace


	@to_replace.setter
	def to_replace(self, to_replace):
		if to_replace is None:
			self._to_replace = None
		else:
			self._to_replace = '_' + to_replace


	@property
	def base_patterns(self):
		return self._base_patterns


	@base_patterns.setter
	def base_patterns(self, base_patterns):
		if base_patterns is None:
			self._base_patterns = []
		elif self.to_replace is None:
			self._base_patterns = [p for p in base_patterns if '_' not in p]
		else:
			self._base_patterns = [p for p in base_patterns if self.to_replace in p]


	def add_prefixes(self, original, prefixes):
		added = [' '.join([pfx, item]) for pfx in prefixes for item in original]
		return added


	def repl_in_base(self, repl_list):
		if self.to_replace is None:
			return self.base_patterns
		old = re.compile(self.to_replace)
		repl = [old.sub(new, item) for new in repl_list for item in self.base_patterns]
		return repl


	def build(self, repl_list=None, prefixes=None):
		if repl_list is None:
			return self.base_patterns
		if prefixes is not None:
			repl_list = self.add_prefixes(repl_list, prefixes)
		patterns = self.repl_in_base(repl_list)
		return list(set(patterns))


class Diagnoser:
	
	def __init__(self, disorder_list, patterns, synonym_map, common_prefixes):
		self.disorder_list = disorder_list
		self.patterns = patterns
		self.synonym_map = synonym_map
		self.common_prefixes = common_prefixes
		self.all_matches = self.get_all_matches()


	def get_all_matches(self):
		matches = self.build_matches(
			to_replace=None,
			synonyms=None,
			prefixes=None
		)
		matches += self.build_matches(
			to_replace='doctor',
			synonyms='doctor',
			prefixes=self.common_prefixes
		)
		disorder_matches = self.build_matches(
			to_replace='condition',
			synonyms=self.disorder_list,
			prefixes=None
		)
		return {dis: matches + match for dis, match in disorder_matches}


	def build_matches(self, to_replace=None, synonyms=None, prefixes=None):
		if isinstance(synonyms, (list, tuple, set)):
			matches = [self.build_matches(to_replace, s, prefixes) for s in synonyms]
			return zip(synonyms, matches)

		matcher = PatternBuilder(to_replace, self.patterns)
		repl_synonyms = self.synonym_map.get(synonyms)
		return matcher.build(repl_synonyms, prefixes)


	def contains_diagnosis(self, comment, disorder):
		for match in self.all_matches.get(disorder, []):
			if match in comment:
				return True
		return False


	def get_diagnoses(self, df):
		users = df.users.to_numpy()
		comments = df.text.to_numpy()
		disorders = df.label_names.to_numpy()

		data = zip(users, comments, disorders)
		data_to_check = tqdm(data, desc='Diagnosing', total=df.shape[0])

		diagnosed_users = set()
		for user, comment, disorder in data_to_check:
			if self.contains_diagnosis(comment, disorder):
				diagnosed_users.add(user)
				
		return list(diagnosed_users)

