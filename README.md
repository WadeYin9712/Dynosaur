
<p align="center" width="100%">
</p>

<div id="top" align="center">
<img src=imgs/dynosaur.png width=150 />

Dynosaur: A Dynamic Growth Paradigm for Instruction-Tuning Data Curation
-----------------------------
<h3> |<a href="https://arxiv.org/abs/2305.14327"> Paper </a> | 
<a href="https://dynosaur-it.github.io/"> Project Website </a> |
<a href="https://huggingface.co/datasets?search=dynosaur"> 🤗 Data </a> |  
<a href="https://huggingface.co/models?sort=trending&search=dynosaur"> 🤗 Model </a> |
</h3>
<h4>
  <a href="https://wadeyin9712.github.io/">Da Yin</a>*, <a href="https://xxxiaol.github.io/">Xiao Liu</a>*, <a href="https://fanyin3639.github.io/">Fan Yin</a>*, <a href="https://maszhongming.github.io/">Ming Zhong</a>*, <a href="https://sites.google.com/view/hbansal">Hritik Bansal</a>, <a href="http://hanj.cs.illinois.edu/">Jiawei Han</a>, <a href="http://web.cs.ucla.edu/~kwchang/">Kai-Wei Chang</a>
</h4>
</div>

Dynosaur aims to 1) build a dynamically growing instruction tuning dataset without low cost for maintenance, and 2) provide a venue to study how to dynamically improve instruction tuning models. The repo contains:

- Dynosaur data
- Crawling code for capturing dataset metadata information
- Instruction generation process with ChatGPT based on metadata
- Model checkpoints

**Usage and License Notices**: All the generated task instructions (except the instances of each task) are released under Apache-2.0 license. The instances of each tasks are subject to the license under which the original dataset was released. These license information are available in Dynosaur data and [`instruction_data/license_info.json`](./instruction_data/license_info.json).

## Updates
- Jul 6, 2023: Dynosaur v1 is coming! `Dynosaur-full` and `Dynosaur-sub-superni` are released at 🤗 [Huggingface Datasets](https://huggingface.co/datasets?search=dynosaur). T5-3B and LLAMA-7B fine-tuned with Dynosaur are released at 🤗 [Huggingface Models](https://huggingface.co/models?sort=trending&search=dynosaur).
- May 23, 2023: Dynosaur v0 is here! Dynosaur-full data, metadata collection method and instruction generation code are released! Will upload them to Huggingface soon!

## Overview

We propose Dynosaur, a large-scale instruction tuning dataset obtained automatically with significantly lower generation costs. Dynosaur leverages the metadata of existing NLP datasets to generate task instructions and organize corresponding inputs/outputs. By utilizing LLMs, we generate multiple task instructions applicable to various NLP domains and determine the relevant data fields for constructing instruction tuning data.

Dynosaur offers several advantages, including 
- Lower generation costs ($11.5 for generating 800K instruction tuning data)
- Decent quality of instruction tuning data (better performance than Alpaca and Instruction GPT-4 on Super-NI)
- Ability to grow dynamically by incorporating new datasets from Huggingface Datasets Platform

## Data Release

We offer [`dynosaur-full`](https://huggingface.co/datasets/Dynosaur/dynosaur-full), containing all the generated instruction tuning data in Dynosaur. It covers most licensed and non-null English datasets in 🤗 Huggingface Datasets as of Feb 23, 2023.
The data is a dictionary containing the following keys:

- `instruction`: `str`, describes task instructions 
- `input`: `str`, input text for the task
- `output`: `str`, ground-truth output text for the task and input text
- `license`: `str`, license of the source dataset
- `website`: `str`, Huggingface website url of the source dataset
- `taskname`: `str`, Huggingface dataset name

We also provide the collected metadata [`huggingface_metadata.jsonl`](https://dynosaur.s3.us-west-1.amazonaws.com/huggingface_metadata.jsonl) and data [`huggingface_data.jsonl`](https://dynosaur.s3.us-west-1.amazonaws.com/huggingface_data.jsonl) from Huggingface. They are the very foundation of synthesizing Dynosaur instructions.

## Data Generation Process

To generate `Dynosaur-full`, please follow the following commands step by step:

### Metadata Collection
```
python parse_hf.py                                            # crawl Huggingface datasets and collect metadata
python license_info.py                                        # capture license information of each dataset
python check_multilingual.py                                  # select English-only datasets
```
If you want to skip these steps, you may use the collected metadata [`huggingface_metadata.jsonl`](https://dynosaur.s3.us-west-1.amazonaws.com/huggingface_metadata.jsonl) and data [`huggingface_data.jsonl`](https://dynosaur.s3.us-west-1.amazonaws.com/huggingface_data.jsonl) on AWS S3.

### Instruction Generation and Filtering
```
python generate_tasks_with_description.py                     # generation description-aware tasks
python generate_tasks_without_description.py                  # generation description-unaware tasks
python filter_invalid_tasks.py                                # filter out invalid tasks
python organize_data.py                                       # organize instruction data
```
These result in the instruction data [`instruction_data/instruction-full.json`](./instruction_data/instruction-full.json) and instruction tuning dataset [`dynosaur-full`](https://huggingface.co/datasets/Dynosaur/dynosaur-full). We will provide the sampled data for evaluation on Super-NI and user instructions soon.

## Fine-tuning

We fine-tune T5-3B and LLaMA-7B with the following hyperparameters:

| Hyperparameter | T5-3B    | LLaMA-7B  |
|----------------|----------|-----------|
| Batch size     | 16       | 128       |
| Learning rate  | 1e-5     | 3e-4      |
| Epochs         | 2        | 3         |
| Max length     | 512      | 512       |

To fine-tune your own models, please refer to the training code of [Tk-Instruct](https://github.com/yizhongw/Tk-Instruct) and [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) or [Alpaca-LORA](https://github.com/tloen/alpaca-lora). 

## Limitations

Dynosaur is still under development and needs a lot of improvements. We are still studying method that can better control the instruction quality and generate more diverse instructions. We are also trying to control any biases introduced in Dynosaur. Stay tuned!

### Citation

If you find this work is relevant with your research, please feel free to cite our work!
```
@article{yin2023dynosaur,
  title={Dynosaur: A Dynamic Growth Paradigm for Instruction-Tuning Data Curation},
  author={Yin, Da and Liu, Xiao and Yin, Fan and Zhong, Ming and Bansal, Hritik and Han, Jiawei and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2305.14327},
  year={2023}
}
```

As our Dynosaur data is based on Huggingface Datasets Resources, please also cite Huggingface Datasets paper:
- Datasets: A Community Library for Natural Language Processing. Lhoest et al., 2021. EMNLP 2021 demo.

### Acknowledgements

We greatly appreciate Huggingface for their great effort in open-sourcing fantastic NLP datasets! We also sincerely thank the authors of all the datasets we incorporate in Dynosaur.

We also thank Yizhong Wang for providing the code for diversity analysis plot and Tk-instruct training code, and Stanford Alpaca Team for releasing the code to finetune LLAMA.
