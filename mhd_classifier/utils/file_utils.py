import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from pathlib import Path, PurePath
from collections import defaultdict
from matplotlib.figure import Figure


class FileManager:
	'''
	Class to handle all file I/O related tasks.
	Used in various classes order to read/write processed files and to allow
	for easy access of project resources.
	'''

	def __init__(self, root=None):
		self.root = root # Root directory to use for reading/writing
		self.dir_files = defaultdict(list) # Groups read files by their containing folder
		self.file_data = defaultdict(lambda: None) # Stores list of files read


	@property
	def root(self):
		return self._root


	@root.setter
	def root(self, root):
		if root is None:
			self._root = Path()
		else:
			self._root = Path(root)
	

	def read(self, path, name=None):
		''' Reads a file from a directory given a directory and file name '''
		# Build the path to read from, relative to our root
		path = self.build_path(path)
		file = File(path=path, full_name=name)
		# If reading a directory, read all files in it
		if name is None and file.full_path.is_dir():
			return [self.read(path, child) for child in listdir(path)]
		file.read()
		self.add(file)
		return self.get(name)


	def write(self, name, data, path=None, infer_extension=True):
		''' Writes a file to a directory given a directory, file name, and data'''
		path = self.build_path(path)
		file = File(path=path, full_name=name, data=data)
		file.write(infer_extension=infer_extension)
		self.add(file)


	def build_path(self, path=None):
		''' Build and return a path, relative to our root.'''
		if path is None:
			return self.root
		return self.root / path


	def add(self, file):
		''' Add a File obj to our managed files. Can be retrieved by get(). '''
		# Cotaining folder and full file name
		folder = file.containing_folder
		name = str(file.full_name)
		# Add (folder, name) and (name, data) (key, value) pairs
		self.dir_files[folder].append(name)
		self.file_data[name] = pickle.loads(pickle.dumps(file))


	def get(self, name, attrs='data'):
		''' Get a File attribute from a File, based on filename. '''
		file = self.file_data.get(name) or self.file_data.get(Path(name).stem)
		if isinstance(attrs, (list, tuple, set)):
			return tuple(getattr(file, attr, None) for attr in attrs)
		return getattr(file, attrs, None)


	def get_dir(self, name, attrs='data'):
		''' Get a File attribute for all File objs in a directory. '''
		files = self.dir_files[name]
		names = [self.get(f, 'name') for f in files]
		return dict(zip(names, [self.get(f, attrs) for f in files]))


class File:
	# Map for file extension and default (expected) object
	ext_obj_map = {
		'.csv': pd.DataFrame(),
		'.npy': np.array([]),
		'.png': Figure(),
		'.txt': [],
	}
	# Extensions and object types valid for file I/O
	io_extensions = tuple(ext_obj_map.keys())
	io_types = tuple(type(obj) for obj in ext_obj_map.values())

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
		if isinstance(full_name, (str, Path, PurePath)):
			self._full_name = Path(full_name)
		else:
			self._full_name = Path()


	@property
	def data(self):
		return self._data


	@data.setter
	def data(self, data):
		# If the file has no data, initialize data based on extension
		if data is None:
			self._data = self.ext_obj_map.get(self.extension, None)
		# If the file's data is a valid I/O type use it as-is
		elif isinstance(data, self.io_types):
			self._data = data
		# If we have invalid I/O data, convert it to a str and add to a list
		else:
			self._data = [str(data)]

	
	def read(self):
		''' Reads a (valid) file based on its extension '''
		if self.readable():
			if self.extension == '.csv':
				self.data = pd.read_csv(self.full_path).fillna('')
			elif self.extension == '.npy':
				self.data = np.load(self.full_path)
			elif self.extension == '.png':
				self.data = plt.imread(self.full_path)
				self.infer_and_update_extension()
			else:
				with open(self.full_path) as txt_file:
					self.data = [line.rstrip() for line in txt_file]


	def write(self, infer_extension=True):
		''' Writes file's data based on its (possibly inferred) extension '''
		if infer_extension:
			self.infer_and_update_extension()
		if self.writeable():
			if self.extension == '.csv':
				self.data.to_csv(self.full_path, index=False)
			elif self.extension == '.npy':
				np.save(self.full_path, self.data)
			elif self.extension == '.png' and hasattr(self.data, 'savefig'):
				self.data.savefig(self.full_path)
			else:
				with open(self.full_path, 'wt') as txt_file:
					txt_file.write('\n'.join(self.data))


	def readable(self):
		''' File is readable if it has a valid extension and it exists '''
		return self.extension in self.io_extensions and self.full_path.exists()


	def writeable(self):
		''' File data is writeable if  '''
		return isinstance(self.data, self.io_types)


	def infer_and_update_extension(self):
		''' Infers a file's extension based on its data type '''
		type_to_ext = {type(o): ext for ext, o in self.ext_obj_map.items()}
		inferred_extension = type_to_ext.get(type(self.data), self.extension)
		self.update_extension(new_extension=inferred_extension)


	def update_extension(self, new_extension):
		''' Updates a file's extension where necessary '''
		self.full_name = self.full_name.with_suffix(new_extension)
		self.full_path = self.path / self.full_name
		self.extension = self.full_name.suffix
		self.data = self.data

