import urllib.request
import re
import json
import time
import sys
import func_timeout
import datasets
from tqdm import tqdm
from func_timeout import func_set_timeout
from datasets import load_dataset, get_dataset_config_names

def check_vision_dataset(data_fields):
	for k in data_fields:
		if "image" in k or "audio" in k:
			return True
	return False

@func_set_timeout(20)
def get_samples(dataset):
	subtask_samples = []
	splits = list(dataset.keys())
	start_time = time.time()
	for split in splits:
		if "train" in split or "validation" in split:
			count = 0
			label_info = dataset[split].features
			fields_to_replace = []
			for label in label_info:
				if isinstance(label_info[label], datasets.features.features.ClassLabel):
					fields_to_replace.append(label)
			tot = 0
			iter_data = iter(dataset[split])
			while True:
				try:
					d = next(iter_data)
					for label in fields_to_replace:
						d[label] = label_info[label].names[d[label]]
					subtask_samples.append(d)
					end_time = time.time()
					if end_time - start_time > 18:
						return subtask_samples
					tot += 1
					if tot == 100:
						break
				except Exception as e:
					return subtask_samples

	return subtask_samples

@func_set_timeout(20)
def load_from_hf(x, config=None):
	if config is None:
		return load_dataset(x, streaming=True)
	else:
		return load_dataset(x, config, streaming=True)

def load_data(x):
	if "***" not in x:
		try:
			configs = get_dataset_config_names(x)
			failed_retry = False
		except Exception as e:
			return {}, []
	else:
		configs = [x.split("***")[1]]
		failed_retry = True
		x = x.split("***")[0]
	x_samples = dict()
	failed_subtask = []

	if len(configs) > 1 or failed_retry:
		all_count = 0
		for i, config in enumerate(configs):
			print(config)
			if i == 10 and all_count == 10:
				break
			try:
				dataset = load_from_hf(x, config)
			except func_timeout.exceptions.FunctionTimedOut:
				all_count += 1
				failed_subtask.append(x+'***'+config)
				print(1)
				print("******")
				continue
			except Exception as e:
				all_count += 1
				print(2)
				print("******")
				continue

			data_fields = []
			if "train" in dataset:
				if dataset['train'].features is not None:
					data_fields = list(dataset['train'].features.keys())
			elif "validation" in dataset:
				if dataset['validation'].features is not None:
					data_fields = list(dataset['validation'].features.keys())
			else:
				splits = list(dataset.keys())
				if dataset[splits[0]].features is not None:
					data_fields = list(dataset[splits[0]].features.keys())

			if len(data_fields) <= 1:
				all_count += 1
				print(3)
				print("******")
				continue

			if check_vision_dataset(data_fields):
				all_count += 1
				print(4)
				print("******")
				continue

			try:
				x_samples[config] = get_samples(dataset)
				if len(x_samples[config]) > 0:
					print("# of collected data for {}:".format(x+'-'+config), len(x_samples[config]))
			except func_timeout.exceptions.FunctionTimedOut:
				all_count += 1
				failed_subtask.append(x+'***'+config)
				print(5)
				print("******")
				continue
			except Exception as e:
				all_count += 1
				print(5)
				print("******")
				continue
	else:
		try:
			dataset = load_from_hf(x)
		except func_timeout.exceptions.FunctionTimedOut:
			print(1)
			print("******")
			return {}, failed_subtask
		except Exception as e:
			print(2)
			print("******")
			return {}, failed_subtask

		if "train" in dataset:
			try:
				data_fields = list(dataset['train'].features.keys())
			except Exception as e:
				return {}, failed_subtask
		elif "validation" in dataset:
			try:
				data_fields = list(dataset['validation'].features.keys())
			except Exception as e:
				return {}, failed_subtask
		else:
			try:
				splits = list(dataset.keys())
				data_fields = list(dataset[splits[0]].features.keys())
			except Exception as e:
				return {}, failed_subtask

		if len(data_fields) <= 1:
			print(3)
			print("******")
			return {}, failed_subtask

		if check_vision_dataset(data_fields):
			print(4)
			print("******")
			return {}, failed_subtask

		try:
			x_samples["orig"] = get_samples(dataset)
			if len(x_samples["orig"]) > 0:
				print("# of collected data for {}:".format(x), len(x_samples["orig"]))
		except func_timeout.exceptions.FunctionTimedOut:
			failed_subtask.append(x)
			print(5)
			print("******")
			return {}, failed_subtask
		except Exception as e:
			print(5)
			print("******")
			return {}, failed_subtask
	
	time.sleep(3)

	return x_samples, failed_subtask
