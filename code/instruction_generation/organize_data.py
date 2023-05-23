import os
import random
import json
import pickle as pkl
from tqdm import tqdm

source_file = '../../instruction_data/huggingface_data.jsonl'
all_data = {}
with open(source_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        all_data[data['task_name']] = data


with open('../../instruction_data/raw_instructions_full.json', 'r') as f:
    all_instructions = json.load(f)

all_instructions_add_label_space = []
for idx, i in enumerate(all_instructions):
    cur_data_original = all_data[i['task_name']]['selected_data'][:100]
    
    flag = True
    label_types = []
    for d in cur_data_original:
        if isinstance(d[i['output_field'][0]], (str, int, float, bool)):
            label_types.append(str(d[i['output_field'][0]]).strip())
        else:
            flag = False
            break
    if not flag:
        continue
    dedup_label_types = list(set(label_types))
    if len(dedup_label_types) <= 10:
        max_num = -1
        count_types = dict()
        flag_label_len = True
        for label in label_types:
            if len(label) > 15:
                flag_label_len = False
            if label not in count_types:
                count_types[label] = 1
            else:
                count_types[label] += 1
        for label in count_types:
            if count_types[label] > max_num:
                max_num = count_types[label]

        if (len(dedup_label_types) == 2 and max_num <= 70) or (len(dedup_label_types) > 2 and max_num <= 50):  # not too imbalanced
            if flag_label_len:
                cur = i.copy()
                cur['instruction'] += f" Answers must be one of {', '.join(dedup_label_types)}."
                all_instructions_add_label_space.append(cur)
            else:
                all_instructions_add_label_space.append(i)
    else:
        all_instructions_add_label_space.append(i)

print(len(all_instructions_add_label_space))


def concat_fields(cur_instance, target_fields, input_or_output):
    if input_or_output == "output":
        if isinstance(cur_instance[target_fields[0]], (str, int, float, bool)):
            return str(cur_instance[target_fields[0]]).strip()
        else:  # list or dict
            return False  # abandon
    else:
        input_j = ""
        for field in target_fields:
            if isinstance(cur_instance[field], (str, int, float, bool)):
                if len(target_fields) > 1:
                    input_j += f"{field} is '" + str(cur_instance[field]).strip() + "'. "
                else:
                    input_j = str(cur_instance[field]).strip()
            elif isinstance(cur_instance[field], list):
                if len(cur_instance[field]) > 0 and isinstance(cur_instance[field][0], str) and None not in cur_instance[field]:
                    if "choice" in field:
                        for i, choice in enumerate(cur_instance[field]):
                            input_j += ' ' + chr(i+ord('A')) + ". '" + choice + "',"
                        input_j = input_j[:-1] + '.'
                    else:
                        input_j += f"'" + "', '".join(cur_instance[field]) + "'. "
                else:
                    return False
            else:
                return False
        return input_j


task_num = 0
all_data_instances = []
valid_instructions = []
for idx, i in enumerate(all_instructions_add_label_space):
    cur_data_original = all_data[i['task_name']]['selected_data']
    input_fields = i['input_fields']
    output_field = i['output_field']
    data_length = []
    instances = []
    for idxj, j in enumerate(cur_data_original):
        input_j = concat_fields(j, input_fields, "input")
        output_j = concat_fields(j, output_field, "output")
        if output_j == False:
            break
        if input_j:
            instances.append({
                'instruction': i["instruction"],
                'input': input_j[:2000],
                'output': output_j[:2000],
                'license': all_data[i["task_name"]]["license"],
                'website': all_data[i["task_name"]]["website"]
            })
    if len(instances) == 0:
        continue
    for instance in instances:
        all_data_instances.append(instance)
    task_num += 1
    valid_instructions.append(i)

print(task_num)
print(len(all_data_instances))

with open('../../instruction_data/instructions-full.json', 'w') as f:
    json.dump(valid_instructions, f, indent=4)

with open('../../instruction_data/dynosaur-full.json', 'w') as f:
    for d in all_data_instances:
        f.write(json.dumps(d)+'\n')