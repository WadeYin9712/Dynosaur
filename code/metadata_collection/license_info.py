import os
import urllib.request
import re
import json
import func_timeout
from tqdm import tqdm
from check_validity import *
from datasets import get_dataset_config_names
from func_timeout import func_set_timeout

def valid_task():
    url = 'https://datasets-server.huggingface.co/valid'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req)
    html = res.read().decode('utf-8').replace("\n", "").replace("\t", "")
    tasks = re.findall(r'"(.*?)",', html)

    return tasks

@func_set_timeout(20)
def get_configs(x):
    try:
        return get_dataset_config_names(x)
    except Exception as e:
        return []

FULL_VALID_DATASET_LIST = valid_task()

url = 'https://huggingface.co/datasets?p=0&sort=downloads'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req)
html = res.read().decode('utf-8')

last_page_index = int(re.findall(r'<a class="rounded-lg px-2.5 py-1  hover:bg-gray-50 dark:hover:bg-gray-800" href=".*?">(.*?)</a>', html)[-1])
license_info = dict()

with open("../instruction_data/license_info.json", 'w') as f:
    for i in tqdm(range(last_page_index)):
        url = 'https://huggingface.co/datasets?p={}&sort=downloads'.format(str(i))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        html = res.read().decode('utf-8')
        dataset_names = re.findall(r'<h4 class="text-md truncate font-mono text-black dark:group-hover:text-yellow-500 group-hover:text-red-600 ">(.*?)</h4>', html)
        all_failed_subtask = []
        for j, dataset_name in enumerate(dataset_names):
            url_dataset = 'https://huggingface.co/datasets/{}'.format(dataset_name)
            req = urllib.request.Request(url_dataset, headers={'User-Agent': 'Mozilla/5.0'})
            html_dataset = urllib.request.urlopen(req).read().decode('utf-8').replace("\n", "").replace("\t", "").replace("&quot;", "")
            html_license = re.findall(r'<a class="tag  tag-white rounded-full" href="/datasets\?license=license:(.*?)">', html_dataset)

            try:
                configs = get_configs(dataset_name)
                if len(html_license) == 0:
                    license_info[dataset_name] = ""
                    f.write(json.dumps({'dataset_name': dataset_name, 'license': "", "website": url_dataset})+'\n')
                else:
                    license_info[dataset_name] = html_license[0]
                    f.write(json.dumps({'dataset_name': dataset_name, 'license': html_license[0], "website": url_dataset})+'\n')
                    if len(configs) != 0:
                        print(dataset_name, configs[0])
                for config in configs:
                    if len(html_license) == 0:
                        license_info[dataset_name+'-'+config] = ""
                        f.write(json.dumps({'dataset_name': dataset_name+'-'+config, 'license': "", "website": url_dataset})+'\n')
                    else:
                        license_info[dataset_name+'-'+config] = html_license[0]
                        f.write(json.dumps({'dataset_name': dataset_name+'-'+config, 'license': html_license[0], "website": url_dataset})+'\n')
            except func_timeout.exceptions.FunctionTimedOut:
                continue
            except Exception as e:
                continue

not_allowed_license = ["other", "unknown", "cc-by-nc-nd-4.0", "cc-by-nd-4.0", "cc-by-nc-nd-3.0", "ofl"]
license_info = dict()
with open("../../instruction_data/license_info.json", 'r') as f:
    for d in f:
        json_d = json.loads(d)
        license_info[json_d["dataset_name"]] = [json_d["license"], json_d["website"]]

tot_tasks = 0
tot_instances = 0
fns = os.listdir('hf_data')
for i, fn in enumerate(fns):
    data = []
    with open(os.path.join('hf_data', fn), 'r') as f:
        for d in f:
            dataset_name = json.loads(d)["task_name"]
            new_d = json.loads(d).copy()
            if dataset_name in license_info:
                new_d["license"] = license_info[dataset_name][0]
                if new_d["license"] in not_allowed_license:
                    continue
                new_d["website"] = license_info[dataset_name][1]
            else:
                continue
            data.append(new_d)
            tot_instances += len(new_d["selected_data"])
        print(fn, len(data))
        tot_tasks += len(data)
        print(tot_tasks)

    with open(os.path.join('hf_data', fn), 'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')
print(tot_instances)

