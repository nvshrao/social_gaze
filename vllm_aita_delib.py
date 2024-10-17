# +
import os
import sys
import torch
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import pickle
import argparse
from collections import Counter
from datasets import load_dataset
import pandas as pd

parser = argparse.ArgumentParser(description='Process model name.')
parser.add_argument('model_short_name', type=str, help='Short name of the model')
args = parser.parse_args()
model_short_name = args.model_short_name

if model_short_name == "llama13":
    model_name = "ondemand_feb24/llama/llama-2-13b-chat-hf"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model=model_name, download_dir='/playpen-ssd/anvesh/cache_dir/')
elif model_short_name == "llama3":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name =  'meta-llama/Meta-Llama-3-8B-Instruct'
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)
elif model_short_name == "llama3.1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)
elif model_short_name == "olmo-7b-0724-sft-dpo":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_name =  "allenai/OLMo-7B-0724-Instruct-hf"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=3)
elif model_short_name == "olmo-7b-0724-sft":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_name =  "allenai/OLMo-7B-0724-SFT-hf"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=3)
elif model_short_name == "mistral-oh":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name =  "teknium/OpenHermes-2.5-Mistral-7B"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=3)
elif model_short_name == "mistral-oh-dpo":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name =  "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=3)
elif model_short_name == "gemma-2-9b-it":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    model_name = "google/gemma-2-9b-it"
    llm = LLM(model=model_name,dtype=torch.bfloat16,gpu_memory_utilization =0.85,tensor_parallel_size = 2,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)
elif model_short_name == "phi-3-med":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model_name = "microsoft/Phi-3-medium-4k-instruct"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)
elif model_short_name == "mixtral":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_name =  'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = LLM(model=model_name, dtype=torch.bfloat16,gpu_memory_utilization =0.75,tensor_parallel_size = 4, download_dir='/playpen-ssd/anvesh/cache_dir/')
elif model_short_name == "llama70q":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_name =  "TheBloke/Llama-2-70b-Chat-AWQ"
    llm = LLM(model=model_name, quantization="AWQ", download_dir='/playpen-ssd/anvesh/cache_dir/')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_name =  "ondemand_feb24/llama/llama-2-70b-chat-hf"
    llm = LLM(model=model_name, dtype=torch.bfloat16, gpu_memory_utilization =0.75,tensor_parallel_size = 4, download_dir='/playpen-ssd/anvesh/cache_dir/')
    
tokenizer = AutoTokenizer.from_pretrained(model_name)


def convert_message_to_prompt(message,tokenize=False):
    if tokenize==True:
        return tokenizer.apply_chat_template(message, tokenize=False)
    else:
        formatted_conversation = ''
        # Loop through each message in the list
        for msg in message:
            # Check if the role is 'user' or 'assistant' and add their content to the string
            if msg['role'] in ['user', 'assistant']:
                formatted_conversation += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

        # Determine the role of the last entry and append an empty line for the opposite role with a space after the colon
        if message[-1]['role'] == 'user':
            formatted_conversation += 'Assistant: '
        elif message[-1]['role'] == 'assistant':
            formatted_conversation += 'User: '
        return formatted_conversation

def load_aita_delib(posts,previous_answers,steps):
    formatted_questions = []
    for idx in np.arange(len(posts)):
        prompt_template = steps[0]
        if model_short_name == "mixtral" or "gemma" in model_short_name or "mistral" in model_short_name:# no system prompt in mixtral
            message = [
            {
                "role": "user", 
                "content":  posts[idx] + "\n\n" +prompt_template
            }
            ]
        else:
            message = [
            {
                "role": "system", 
                "content": ""
            },
            {
                "role": "user", 
                "content":  posts[idx] + "\n\n" +prompt_template
            }
            ]
        for i in np.arange(len(previous_answers[idx])):
            message.extend(
                        [{
            "role": "assistant", 
            "content":  previous_answers[idx][i]
                        },
                        {
            "role": "user", 
            "content":  steps[i+1]
                        }]
            )
        message = convert_message_to_prompt(message,tokenize=True)
        formatted_questions.append(message)
    return formatted_questions

steps =     [
        "Quickly summarize the narrative.",
        "Highlight the narrator’s actions or decisions that are relevant to the situation.",
        "Highlight the actions, decisions, or responses of other people involved that are relevant to the situation.",
        "Given these actions and contexts, make a decision. State explicitly, whether the narrator alone is at fault (YTA), the narrator is not at fault (NTA). Start with your decision, followed by a concise supporting rationale."]


df =pd.read_csv("./data/evaluate_rel_with_score.csv")
df=df.filter(['flair', 'text', 'label','comment', 'Upvote', 'Upvote_label', 'base_model_generations','rl_model_generations'])
posts = df["text"]

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def extract_label(input_txt):
    label_pattern = r'\b(?:YTA|NTA)\b'
    match = re.search(label_pattern, input_txt)
    label = match.group(0) if match else 'NA'
    return label

for iteration in np.arange(5):
  previous_answers =[[] for _ in range(len(posts))]
  print("iter",iteration)
  for step in np.arange(len(steps)):# 0,1,2,3
      print("step",step)
      folder_path = os.path.join("results","delib",model_short_name)
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)
      cache_path = os.path.join(folder_path,f"{model_short_name}_step_{step}_cache_{iteration}.pickle")
      # Check if cache exists for the current step
      if not os.path.exists(cache_path):
          new_data = load_aita_delib(posts,previous_answers,steps)
          all_outputs_delib = []
          for X in tqdm(batch(new_data, 1600)):
              outputs = llm.generate(X,SamplingParams(top_k=10,max_tokens=400))
              all_outputs_delib.append([o.outputs[0].text.split("<|eot_id|>")[0] for o in outputs])
          all_outputs_delib = list(itertools.chain.from_iterable(all_outputs_delib))#remove batches
          [i.append(j) for i,j in zip(previous_answers,all_outputs_delib)]
          with open(cache_path, "wb") as cache_file:
              pickle.dump(previous_answers, cache_file)
          print(f"Cache saved for step {step} at {cache_path}")
      else:
          previous_answers = pickle.load(open(cache_path,"rb"))
