import openai
import json
import os
import time
import logging
import backoff
import random
from tqdm import tqdm
import pickle as pkl
import sys

logging.basicConfig(level=logging.INFO)

openai.api_key = os.getenv("OPENAI_API_KEY")

source_file = '../../instruction_data/huggingface_metadata.jsonl'
data = []
with open(source_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))
print(len(data), data[0].keys())
        
def truncate(example, max_len):  # remove nested fields and truncate every field to max_len chars
    truncated = {}
    for i in example:
        if 'id' in i:
            continue
        if isinstance(example[i], (str, int, float, bool)):
            truncated[i] = str(example[i]).strip()[:max_len]
        else:
            not_nested = True
            if isinstance(example[i], list):
                cur = []
                for j in example[i]:
                    if isinstance(j, str):
                        tmp = cur.copy()
                        tmp.append(j)
                        if len(json.dumps(tmp)) > max_len:
                            break
                        else:
                            cur.append(j)
                    else:
                        not_nested = False
                        break
            elif isinstance(example[i], dict):
                cur = {}
                for j in example[i]:
                    if isinstance(example[i][j], (str, int, float, bool)):
                        tmp = cur.copy()
                        tmp[j] = example[i][j]
                        if len(json.dumps(tmp)) > max_len:
                            break
                        else:
                            cur[j] = example[i][j]
                    else:
                        not_nested = False
                        break
            else:  # NoneType
                continue
            if not_nested:
                truncated[i] = cur
        
    return truncated


def truncate_metadata(metadata, max_len):
    truncated = {}
    for i in metadata:
        if i in ['multilinguality', 'data_fields', 'supported_task', 'task_category', 'task_ids', 'summary']:
            continue
        if i != 'selected_data':
            truncated[i] = metadata[i][:max_len]
        else:
            truncated[i] = [truncate(j, 1000) for j in metadata['selected_data'][:2]]
    return truncated

demonstrations = {'squad': {'data': data[761], 'tasks': {
    'task1': {'instruction': 'Please answer the question based on the Wikipedia article. The answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.', 'input_fields':['title', 'context', 'question'], 'output_field':['answers']},
    'task2': {'instruction': 'Create a question provided the article.', 'input_fields':['context'], 'output_field':['question']},
    'task3': {'instruction': 'Can you write a title for the passage?', 'input_fields':['context'], 'output_field':['title']}}},
                  'anli': {'data': data[765], 'tasks': {
    'task1': {'instruction': 'Natural language inference (NLI) is the task of determining whether a hypothesis is entailment, contradiction, or neutral given a premise.', 'input_fields':['premise', 'hypothesis'], 'output_field':['label']},
    'task2': {'instruction': 'This task asks models to write a premise that entails, contradicts to, or is neutral to the hypothesis.', 'input_fields':['label', 'hypothesis'], 'output_field':['premise']}}},
                 'math_qa':{'data': data[2008], 'tasks': {
    'task1': {'instruction': 'Choose the correct answer for the math problem.', 'input_fields':['Problem', 'options'], 'output_field':['correct']},
    'task2': {'instruction': 'Help to provide the rationale for the question', 'input_fields':['Problem'], 'output_field':['Rationale']},
    'task3': {'instruction': 'Can you design a math problem based on the formula?', 'input_fields':['annotated_formula'], 'output_field':['Problem']}}},
                'common_gen':{'data': data[836], 'tasks': {
    'task1': {'instruction': 'Given a set of common concepts; the task is to construct a coherent sentence describing an everyday scenario using these concepts.', 'input_fields':['concepts'], 'output_field':['target']},
    'task2': {'instruction': 'Extract the key concepts from the sentence.', 'input_fields':['target'], 'output_field':['concepts']}}}}

instruction = 'Given a few examples of a dataset, our goal is to design up to three different tasks based on this dataset. Each task should still be a dictionary, including the instruction, input fields and one output field. The following are two examples.\n\nExample 1:\nInput:\n'
instruction += str(truncate_metadata(demonstrations['math_qa']['data'], 1000))+ '\n\nTasks:\n' + str(demonstrations['math_qa']['tasks']) + '\n\nExample 2:\nInput:\n'
instruction += str(truncate_metadata(demonstrations['common_gen']['data'], 1000)) + '\n\nTasks:\n' + str(demonstrations['common_gen']['tasks'])
instruction += '\n\nNow given a dictionary as input, please help us to generate new tasks. You may stop when there is no more plausible task.\n\nInput:\n'
                   
prompt = {}
                   
for i in data:  
    cur = truncate_metadata(i, 1000)
    if len(cur['selected_data'][0]) <= 1:
        continue
    
    text = instruction
    
    text += str(cur)+ '\n\nNote that the input and output fields should not be duplicated and should both appear in \'selected_data\'. Each task should still be a dictionary, containing no text or explanations outside the dictionary.\n\nTasks:'.replace('\'selected_data\'', str(list(cur['selected_data'][0].keys())))
    
    prompt[i['task_name']] = text
    
    if len(prompt) == 1:
        print(text)
        
print(len(prompt))

system_info = "You are a helpful assistant for task design."

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
def chat_completions_with_backoff(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": system_info}, {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0
    )
    return response

prediction_filename = '../../instruction_data/instruction_wo_description'
prediction = {}
    
bad_cases = 0
for i in tqdm(prompt):
    if i in prediction:
        continue
    try:
        response = chat_completions_with_backoff(prompt[i]).choices[0].message.content.strip()
        logging.info(prompt[i].split('Input:')[-1].strip())
        logging.info(response)
        prediction[i] = response
    except:
        print(i)
        bad_cases += 1
        if bad_cases >= 10:
            sys.exit(1)
        else:
            continue
    
with open(prediction_filename+'.pkl', 'wb') as f:
    pkl.dump([prediction, prompt], f)