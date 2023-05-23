import json
import pickle as pkl
import ast
import argparse

source_file = '../../instruction_data/huggingface_metadata.jsonl'
data = []
with open(source_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

task_data = {}
for i in data:
    task_data[i['task_name']] = i

with open('../../instruction_data/instruction_w_description.pkl', 'rb') as f:
    instruction_w_description, _ = pkl.load(f)
with open('../../instruction_data/instruction_wo_description.pkl', 'rb') as f:
    instruction_wo_description, _ = pkl.load(f)

def filter(instruction_dict):
    preprocessed_inst = []
    cnt_invalid_field = 0
    cnt_overlap_field = 0
    cnt_empty_field = 0
    cnt_parse_fail = 0
    cnt_duplicate_inst = 0

    for i in instruction_dict:
        cur = instruction_dict[i]
        try:
            cur = ast.literal_eval(cur)
        except:
            cnt_parse_fail += 1
            continue

        instructions = [cur[j]['instruction'] for j in cur]
        input_fields = [cur[j]['input_fields'] for j in cur]
        output_fields = [cur[j]['output_field'] for j in cur]

        all_fields = [x for x in task_data[i]['selected_data'][0].keys()]
        
        for p in range(len(input_fields)):
            if len(output_fields[p]) == 0:
                cnt_empty_field += 1
                continue

            f_invalid_field = False
            for j in input_fields[p] + output_fields[p]:
                if not j in all_fields:
                    f_invalid_field = True
                    break
            if f_invalid_field:
                cnt_invalid_field += 1
                continue
                
            f_overlap_field = False
            for j in input_fields[p]:
                for k in output_fields[p]:
                    if j == k:
                        f_overlap_field = True
                        break
            if f_overlap_field:
                cnt_overlap_field += 1
                continue
            if p != 0 and instructions[p] in instructions[:p]:
                cnt_duplicate_inst += 1
                continue

            cur['task'+str(p+1)]["task_name"] = i
            preprocessed_inst.append(cur['task'+str(p+1)])

    print(cnt_parse_fail, cnt_invalid_field, cnt_overlap_field, cnt_empty_field, cnt_duplicate_inst)
    print(len(preprocessed_inst))

    return preprocessed_inst

preprocessed_instruction_w_description = filter(instruction_w_description)
preprocessed_instruction_wo_description = filter(instruction_wo_description)

preprocessed_instruction_all = preprocessed_instruction_w_description + preprocessed_instruction_wo_description
# remove duplicate instructions
preprocessed_instruction_all = [i for n, i in enumerate(preprocessed_instruction_all) if i not in preprocessed_instruction_all[n + 1:]]

with open('../../instruction_data/raw_instructions_full.json', 'w') as f:
    json.dump(preprocessed_instruction_all, f, indent=4)
