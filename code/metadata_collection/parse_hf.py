import os
import urllib.request
import re
import json
from tqdm import tqdm
from remove_tags import *
from check_validity import *
from download_data import *

def valid_task():
    url = 'https://datasets-server.huggingface.co/valid'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    html = res.read().decode('utf-8').replace("\n", "").replace("\t", "")
    tasks = re.findall(r'"(.*?)",', html)

    return tasks

FULL_VALID_DATASET_LIST = valid_task()

def remove_noise(x):
    x = remove_p_tags(x)
    x = remove_href_tags(x)
    x = remove_span_tags(x)
    x = remove_li_tags(x)
    x = remove_font_tags(x)
    x = remove_table_tags(x)
    x = replace_tags(x)
    x = re.sub('\n+', '\n\n', x)

    return x

def convert_to_json(dataset_header, dataset_summ, dataset_supported_task, dataset_data_fields):
    dataset = dict()
    dataset["summary"] = dataset_summ
    dataset["supported_task"] = dataset_supported_task
    dataset["data_fields"] = dataset_data_fields
    dataset["task_category"] = ', '.join(re.findall(r'id:task_categories:(.*?),', dataset_header))
    dataset["task_ids"] = ', '.join(re.findall(r'id:task_ids:(.*?),', dataset_header))

    return dataset

if not os.path.exists("parsed_data"):
    os.mkdir("parsed_data")

url = 'https://huggingface.co/datasets?p=0&sort=downloads'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req)
html = res.read().decode('utf-8')

last_page_index = int(re.findall(r'<a class="rounded-lg px-2.5 py-1  hover:bg-gray-50 dark:hover:bg-gray-800" href=".*?">(.*?)</a>', html)[-1])
dataset_description = dict()
all_task_num = 0
all_task_data_num = 0

for i in tqdm(range(last_page_index)):
    instruct_gen_sample = []
    failed_instruct_gen_sample = []
    with open("parsed_data/instruct_gen_sample_page_{}_samples.jsonl".format(str(i)), "w") as f:
        url = 'https://huggingface.co/datasets?p={}&sort=downloads'.format(str(i))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        html = res.read().decode('utf-8')
        dataset_names = re.findall(r'<h4 class="text-md truncate font-mono text-black dark:group-hover:text-yellow-500 group-hover:text-red-600 ">(.*?)</h4>', html)
        all_failed_subtask = []
        for j, dataset_name in enumerate(dataset_names):
            if dataset_name not in FULL_VALID_DATASET_LIST:
                continue
            print(dataset_name)
            url_dataset = 'https://huggingface.co/datasets/{}'.format(dataset_name)
            req = urllib.request.Request(url_dataset, headers={'User-Agent': 'Mozilla/5.0'})
            html_dataset = urllib.request.urlopen(req).read().decode('utf-8').replace("\n", "").replace("\t", "").replace("&quot;", "")
            html_license = re.findall(r'<span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">License:</span>', html_dataset)
            if not html_license:
                continue
            else:
                html_dataset_header = re.findall(r'<header class=([\s\S]*)</header>', html_dataset)
                html_dataset_card = re.findall(r'<div class="2xl:pr-6">([\s\S]*)</div>', html_dataset)
                if len(html_dataset_header) == 0:
                    continue
                else:
                    html_dataset_header = html_dataset_header[0]
                if len(html_dataset_card) == 0:
                    continue
                else:
                    html_dataset_card = html_dataset_card[0]

                dataset_header = re.findall(r'</h1><div class="SVELTE_HYDRATER contents" data-props="(.*?)" data-target="DatasetHeaderTags"', html_dataset_header)[0].replace("&quot;", "")
                dataset_summ = re.findall(r'Dataset Summary</span></h3>(.*?)<h3', html_dataset_card)
                dataset_supported_task = re.findall(r'Supported Tasks and Leaderboards</span></h3>(.*?)<h3', html_dataset_card)
                dataset_data_fields = re.findall(r'Data Fields</span></h3>(.*?)<h3', html_dataset_card)
                languages = list(set(re.findall(r'id:language:(.*?),', html_dataset)))
                if len(languages) > 1 or (len(languages) == 1 and languages[0] != "en" and languages[0] != "code"):
                    multilingual = "multilingual"
                else:
                    multilingual = "english"
                print(languages, multilingual)

                if len(dataset_summ) != 0:
                    dataset_summ = remove_noise(dataset_summ[0])
                if len(dataset_supported_task) != 0:
                    dataset_supported_task = remove_noise(dataset_supported_task[0])
                if len(dataset_data_fields) != 0:
                    dataset_data_fields = remove_noise(dataset_data_fields[0])

                dataset_json_format = convert_to_json(dataset_header, dataset_summ, dataset_supported_task, dataset_data_fields)
                dataset_description[dataset_name] = dataset_json_format

                dataset_samples, failed_subtask = load_data(dataset_name)
                all_failed_subtask += failed_subtask

                if dataset_samples.keys():
                    for subtask in dataset_samples:
                        if subtask != "orig":
                            selected_data = {"task_name": dataset_name+'-'+subtask, "selected_data": dataset_samples[subtask], "multilinguality": multilingual}
                        else:
                            selected_data = {"task_name": dataset_name, "selected_data": dataset_samples[subtask], "multilinguality": multilingual}
                        all_task_data_num += len(dataset_samples[subtask])
                        instruct_gen_sample.append({**selected_data, **dataset_json_format})
                        f.write(json.dumps(instruct_gen_sample[-1], default=str)+'\n')
                print("----------")
        
        for dataset_name in all_failed_subtask:
            print("FAILED", dataset_name)
            dataset_samples, _ = load_data(dataset_name)
            if dataset_samples.keys():
                dataset_json_format = dataset_description[dataset_name.split("***")[0]]
                for subtask in dataset_samples:
                    if subtask != "orig":
                        selected_data = {"task_name": dataset_name.replace("***", "-"), "selected_data": dataset_samples[subtask], "multilinguality": multilingual}
                    else:
                        selected_data = {"task_name": dataset_name, "selected_data": dataset_samples[subtask], "multilinguality": multilingual}
                    all_task_data_num += len(dataset_samples[subtask])
                    failed_instruct_gen_sample.append({**selected_data, **dataset_json_format})
                    if len(dataset_samples[subtask]) > 0:
                        print("# of collected data for {}:".format(selected_data["task_name"]), len(dataset_samples[subtask]))
                    f.write(json.dumps(failed_instruct_gen_sample[-1], default=str)+'\n')
            print("----------")

        all_task_num += len(instruct_gen_sample) + len(failed_instruct_gen_sample)
        print("# of tasks:", all_task_num)
        print("# of data:", all_task_data_num)

    time.sleep(min(1800, 0.5*(len(instruct_gen_sample)+len(failed_instruct_gen_sample))))
