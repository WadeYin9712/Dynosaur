import json
import os
from lang_list import *

LANG_LIST = [lang[0] for lang in LANG_LIST]

fns = os.listdir('parsed_data')

if not os.path.exists("hf_data"):
    os.mkdir("hf_data")

s = 0
s1 = 0
s_trans = 0
for fn in fns:
	if "instruct_gen_sample_page_" not in fn:
		continue
	with open(os.path.join('parsed_data', fn), "r") as f:
		data = [json.loads(d) for d in f]

	with open("hf_data/"+fn, "w") as f:
		for d in data:
			print(d.keys())
			if len(d["selected_data"]) == 0:
				continue
			dataset_summary = d["summary"]

			if d["multilinguality"] == "english":
				s1 += 1
			elif "paws-x" not in d["task_name"] and ("translat" in dataset_summary or "Translat" in dataset_summary or "bitext" in dataset_summary):
				s_trans += 1
				d["multilinguality"] = "translation"
			else:
				s += 1
				d["multilinguality"] = "multilingual"

			f.write(json.dumps(d)+'\n')
	print(fn, s1, '/', s_trans, '/', s)

tot = 0
fns = os.listdir('hf_data')
with open("../../instruction_data/huggingface_metadata.jsonl", "w") as f:
	for fn in fns:
		with open(os.path.join('hf_data', fn), "r") as f1:
			data = [json.loads(d) for d in f1]
		for i in range(len(data)):
			if data[i]["multilinguality"] == "english":
				d = data[i].copy()
				if len(d["selected_data"]) >= 2:
					d["selected_data"] = d["selected_data"][:2]
				f.write(json.dumps(d)+'\n')
				tot += 1
print(tot)

with open("../../instruction_data/huggingface_data.jsonl", "w") as f:
    for fn in fns:
        with open(os.path.join('hf_data', fn), "r") as f1:
            data = [json.loads(d) for d in f1]
        for i in range(len(data)):
            if data[i]["multilinguality"] == "english":
                d = data[i].copy()
                f.write(json.dumps(d)+'\n')