# +
import os
import sys
import torch
from collections import Counter
import pandas as pd
import itertools
import numpy as np
import argparse
import pickle
import pickle
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
from collections import Counter

parser = argparse.ArgumentParser(description='Process model name.')
parser.add_argument('model_short_name', type=str, help='Short name of the model')
args = parser.parse_args()
model_short_name = args.model_short_name

if model_short_name == "llama13":
    model_name = "ondemand_feb24/llama/llama-2-13b-chat-hf"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model=model_name, download_dir='/playpen-ssd/anvesh/cache_dir/')
  
elif model_short_name == "llama3":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_name =  'meta-llama/Meta-Llama-3-8B-Instruct'
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=3)

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

elif model_short_name == "llama3.1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)

elif model_short_name == "phi-3-med":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_name = "microsoft/Phi-3-medium-4k-instruct"
    llm = LLM(model=model_name,dtype=torch.bfloat16,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)
  
elif model_short_name == "gemma-2-9b-it":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    model_name = "google/gemma-2-9b-it"
    llm = LLM(model=model_name,dtype=torch.bfloat16,gpu_memory_utilization =0.85,tensor_parallel_size = 2,download_dir='/playpen-ssd/anvesh/cache_dir/',seed=4)

elif model_short_name == "mixtral":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_name =  'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = LLM(model=model_name, dtype=torch.bfloat16,gpu_memory_utilization =0.75,tensor_parallel_size = 4, download_dir='/playpen-ssd/anvesh/cache_dir/')
    
elif model_short_name == "llama70q":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_name =  "TheBloke/Llama-2-70b-Chat-AWQ"
    llm = LLM(model=model_name, quantization="AWQ", download_dir='/playpen-ssd/anvesh/cache_dir/')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_name =  "ondemand_feb24/llama/llama-2-70b-chat-hf"
    llm = LLM(model=model_name, dtype=torch.bfloat16,gpu_memory_utilization =0.75,tensor_parallel_size = 4, download_dir='/playpen-ssd/anvesh/cache_dir/')
    
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
    
    
from datasets import load_dataset
import pandas as pd

question_template = """Answer the given question and show your work first.
You must output only the answer in your final sentence like ‘‘Therefore, the answer is ...’’.
Question is {question}"""

# Format questions according to the template
def load_aita():
    formatted_questions = []
    df =pd.read_csv("./rakesh_test_set/evaluate_rel_with_score.csv")
    df=df.filter(['flair', 'text', 'label','comment', 'Upvote', 'Upvote_label', 'base_model_generations','rl_model_generations'])
    for idx, row in df.iterrows():
        #print(prompt_template + "\n\n" + df["text"].iloc[0])
        prompt_template = "Given this narrative, make a decision. State explicitly, whether the narrator alone is at fault (YTA), the narrator is not at fault (NTA). Start with your decision, followed by a concise supporting rationale."
        #print(prompt_template + "\n\n" + df["text"].iloc[0])
      #print(prompt_template + "\n\n" + df["text"].iloc[0])
        post = row["text"]
        if model_short_name == "mixtral" or "gemma" in model_short_name or "mistral" in model_short_name: # no system prompt in mixtral
            message = [
            {
                "role": "user", 
                "content":  post.strip() + "\n\n" +prompt_template
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
                "content":  post.strip() + "\n\n" +prompt_template
            }
            ]
        message = convert_message_to_prompt(message,tokenize=True)
        formatted_questions.append([message,row["label"]])
    return formatted_questions

aita_dataset = load_aita()
def extract_label(input_txt):
    label_pattern = r'\b(?:YTA|NTA)\b'
    match = re.search(label_pattern, input_txt)
    label = match.group(0) if match else 'NA'
    return label
# Example: print the first formatted question and its answer
# print(aita_dataset[1])

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]        

N=5
for iteration in np.arange(N):
  print(iteration)
  file_path = f"./new_data_vllm/baseline/{model_short_name}_{iteration}.pickle"
  if not os.path.exists(file_path):
    all_outputs = []
    for X in tqdm(batch(aita_dataset, 1600)):
        qs = [x[0] for x in X]
        outputs = llm.generate(qs,SamplingParams(top_k=10,max_tokens=300))
        all_outputs.append([o.outputs[0].text for o in outputs])
    # llama 13 b took 15 minutes for 1300 examples
    all_outputs = list(itertools.chain.from_iterable(all_outputs))
    #all_outputs = [i.split("User: ")[0].strip() for i in all_outputs]
    #all_outputs = [i.split("Assistant: ")[0].strip() for i in all_outputs]
    pred = [extract_label(i) for i in all_outputs]
    labels = ["NTA" if i[1]==0 else "YTA" for i in aita_dataset ]
    print(Counter(pred))
    print(classification_report(labels,pred,labels=["NTA","YTA"],digits=4))
    pickle.dump(all_outputs,open(file_path,"wb"))