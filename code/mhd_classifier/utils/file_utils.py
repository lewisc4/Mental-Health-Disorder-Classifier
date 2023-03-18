import os
import pandas as pd
import numpy as np

from copy import deepcopy
from pathlib import Path, PurePath
from collections import defaultdict


class File:
	data_type_map = {
		'.csv': pd.DataFrame(),
		'.npy': np.array([]),
		'.txt': [],
	}

	def __init__(self, path, full_name=None, data=None):
		self.path = path
		self.full_name = full_name
		self.full_path = self.path / self.full_name
		self.containing_folder = PurePath(self.path).name
		self.name = self.full_name.stem
		self.extension = self.full_name.suffix
		self.data = data


	@property
	def path(self):
		return self._path


	@path.setter
	def path(self, path):
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)
		self._path = path
	

	@property
	def full_name(self):
		return self._full_name


	@full_name.setter
	def full_name(self, full_name):
		if isinstance(full_name, str):
			self._full_name = Path(full_name)
		else:
			self._full_name = Path('')


	@property
	def data(self):
		return self._data


	@data.setter
	def data(self, data):
		if data is None:
			self._data = self.data_type_map.get(self.extension, None)
		else:
			self._data = data

	
	def valid_io_ext(self):
		return self.extension in self.data_type_map.keys()


	def readable(self):
		return self.valid_io_ext() and self.full_path.exists()


	def read(self):
		if not self.readable():
			self.data = None
		elif self.extension == '.csv':
			self.data = pd.read_csv(self.full_path).fillna('')
		elif self.extension == '.npy':
			self.data = np.load(self.full_path)
		else:
			with open(self.full_path) as txt_file:
				self.data = [line.rstrip() for line in txt_file]


	def writeable(self):
		if self.valid_io_ext():
			expected_type = type(self.data_type_map[self.extension])
			return isinstance(self.data, expected_type)
		return False


	def write(self):
		if not self.writeable():
			return
		elif self.extension == '.csv':
			self.data.to_csv(self.full_path, index=False)
		elif self.extension == '.npy':
			np.save(self.full_path, self.data)
		else:
			with open(self.full_path, 'wt') as txt_file:
				txt_file.write('\n'.join(self.data))


class FileManager:
	'''
	Class to handle all file I/O related tasks
	Used across various other classes in order to read and write processed files
	'''

	def __init__(self):
		self.dir_files = defaultdict(list) # Groups read files by their containing folder
		self.file_data = defaultdict(lambda: None) # Stores list of files read


	def get(self, name, attrs='data'):
		file = self.file_data.get(name) or self.file_data.get(Path(name).stem)
		if isinstance(attrs, (list, tuple, set)):
			return tuple(getattr(file, attr, None) for attr in attrs)
		return getattr(file, attrs, None)


	def get_dir(self, name, attrs='data'):
		# If we are getting a directory (i.e. all files in a directory)
		files = self.dir_files[name]
		names = [self.get(f, 'name') for f in files]
		return dict(zip(names, [self.get(f, attrs) for f in files]))


	def add(self, file):
		# Cotaining folder and full file name
		folder = file.containing_folder
		name = str(file.full_name)
		# Add (folder, name) and (name, data) (key, value) pairs
		self.dir_files[folder].append(name)
		self.file_data[name] = deepcopy(file)


	def read(self, path, name=None):
		''' Reads a file from a directory given a directory and file name '''
		file = File(path=path, full_name=name)
		# If there is no file name, read all files in the path/dir
		if name is None:
			return [self.read(path, name) for name in os.listdir(path)]
		file.read()
		self.add(file)
		return self.get(name)


	def write(self, path, name, data):
		''' Writes a file to a directory given a directory, file name, and data'''
		file = File(path=path, full_name=name, data=data)
		file.write()
		self.add(file)

