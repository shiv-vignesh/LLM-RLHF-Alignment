import os
import json
import torch
from typing import Iterable
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, LlamaConfig

from src.utils.utils import MODEL_CLASSES, IGNORE_INDEX, PAD_TOKEN_ID

"""
Note:
    - `__getitem__` runs in parallel across DataLoader worker processes when num_workers > 0.
    - Use it for per-sample preprocessing that benefits from multiprocessing (e.g., tokenization, decoding, image augmentations).
"""
"""
Note:
    - `collate_fn` always runs in the main process after workers return their samples.
    - Use it for batch-level operations (e.g., padding, batching, dynamic truncation).
    - Heavy per-sample preprocessing here will not be parallelized and may become a bottleneck.
    
Hybrid Approach:
    - Perform expensive per-sample preprocessing in `Dataset.__getitem__` (parallelized).
    - Perform batch-dependent operations (e.g., padding, mask creation) in `collate_fn`.
    - This approach balances multiprocessing efficiency and batch-level flexibility.
"""


class AnthrophicRLHFDataset(Dataset):
    def __init__(self, jsonl_path:str, 
                data_dir_type:str=None, no_multi_turn:bool=True):
        
        self.data = []
        self.jsonl_path = jsonl_path
        self.data_dir_type = data_dir_type
        self.no_multi_turn = no_multi_turn #for models with limited context length.
        
        self.parse_jsonl(self.jsonl_path)
        
    def parse_jsonl(self, jsonl_path:str):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line = json.loads(line)
                    
                    if "chosen" in line and "rejected" in line:
                        # Check for single-turn: exactly 1 Human and 1 Assistant
                        if self.no_multi_turn:
                            chosen_text = line["chosen"].strip()
                            rejected_text = line["rejected"].strip()
                            
                            num_human_chosen = chosen_text.count("Human:")
                            num_assistant_chosen = chosen_text.count("Assistant:")
                            num_human_rejected = rejected_text.count("Human:")
                            num_assistant_rejected = rejected_text.count("Assistant:")
                            
                            if (num_human_chosen == 1 and num_assistant_chosen == 1 and
                                num_human_rejected == 1 and num_assistant_rejected == 1):
                                self.data.append(line)
                        else:
                            self.data.append(line)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
class AnthrophicRLHFDataCollator(object):

    def __init__(self, model_name:str, 
                max_prompt_length:int=256, 
                max_response_length:int=256):

        self.config:AutoConfig
        self.tokenizer:AutoTokenizer

        if model_name in MODEL_CLASSES:
            config, tokenizer, _ = MODEL_CLASSES[model_name]
        else:
            config, tokenizer, _ = MODEL_CLASSES["auto"]

        self.config = config.from_pretrained(model_name)
        self.tokenizer = tokenizer.from_pretrained(model_name)
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_input_length = self.max_prompt_length + self.max_response_length
        
    def split_dialogue(self, dialogue:str):
        
        parts = dialogue.strip().split("Assistant:")
        if len(parts) < 2:
            raise ValueError("Dialogue missing Assistant part")        
        
        context = "Assistant:".join(parts[:-1]) # for multi-turn conversations, append all context up until the last prompt-response.
        response = parts[-1].strip()
        
        return context.strip(), f'Assistant: {response}'
    
    def truncate(self, encoded_ids:list):

        if len(encoded_ids) > self.max_input_length:
            encoded_ids = encoded_ids[:self.max_input_length - 2]
        
        encoded_ids += [self.tokenizer.eos_token_id]            

        return encoded_ids
    
    def preprocess_sample(self, context:str):
        
        input_ids = self.tokenizer.encode(text=context, add_special_tokens=False)
    
    def preprocess_inference(self, data_items:Iterable[dict]):
        """
        testing dataset for inference mode to compute BLEU score and ROGUE. 

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=AnthrophicRLHFDataCollator(model_name).preprocess_inference
        )

        for batch in dataloader:
            chosen_ids = batch["chosen_context_ids"]
            attention_mask = batch["chosen_attention_mask"]

            # Move to device
            chosen_ids = chosen_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate or compute BLEU/ROUGE
            outputs = model.generate(input_ids=chosen_ids, attention_mask=attention_mask, max_new_tokens=128)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        """

        result = {
            "chosen_contexts":[],
            "chosen_responses":[],
            "rejected_contexts":[],
            "rejected_responses":[],
            "chosen_context_ids":[],
            "rejected_context_ids":[],            
        }
        
        for data_item in data_items:
            chosen_context, chosen_response = self.split_dialogue(data_item["chosen"])
            rejected_context, rejected_response = self.split_dialogue(data_item["rejected"])
            
            chosen_context_ids = self.tokenizer.encode(text=chosen_context, add_special_tokens=False)
            rejected_context_ids = self.tokenizer.encode(text=rejected_context, add_special_tokens=False)

            if len(chosen_context_ids) > self.max_prompt_length - 1:
                chosen_context_ids = chosen_context_ids[:self.max_prompt_length - 1]

            if len(rejected_context_ids) > self.max_prompt_length - 1:
                rejected_context_ids = rejected_context_ids[:self.max_prompt_length - 1]
   
            result["chosen_context_ids"].append(chosen_context_ids)
            result["rejected_context_ids"].append(rejected_context_ids)
            
            result["chosen_responses"].append(chosen_response)
            result["rejected_responses"].append(rejected_response)

        result = {k:torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=self.tokenizer.pad_token_id) for k,v in result.items() if torch.is_tensor(v)}
        result["chosen_attention_mask"] = result["chosen_context_ids"].ne(self.tokenizer.pad_token_id).long()
        result["rejected_attention_mask"] = result["rejected_context_ids"].ne(self.tokenizer.pad_token_id).long()
        
        return result
    
    def __call__(self, data_items:Iterable[dict]):
        """
        [dict_keys(['chosen', 'rejected'])]
        """

        result = {
            "chosen_input_ids": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_labels": [],
        }

        for data_item in data_items:

            chosen_context, chosen_response = self.split_dialogue(data_item["chosen"])
            rejected_context, rejected_response = self.split_dialogue(data_item["rejected"])
            
            chosen_context_ids = self.tokenizer.encode(text=chosen_context, add_special_tokens=False)
            chosen_response_ids = self.tokenizer.encode(text=chosen_response, add_special_tokens=False)
            rejected_context_ids = self.tokenizer.encode(text=rejected_context, add_special_tokens=False)
            rejected_response_ids = self.tokenizer.encode(text=rejected_response, add_special_tokens=False)
            
            if len(chosen_context_ids) > self.max_prompt_length - 1:
                chosen_context_ids = chosen_context_ids[:self.max_prompt_length - 1]
                
            if len(rejected_context_ids) > self.max_prompt_length - 1:
                rejected_context_ids = rejected_context_ids[:self.max_prompt_length - 1]
                
            if len(chosen_response_ids) > self.max_response_length - 1:
                chosen_response_ids = chosen_response_ids[:self.max_response_length - 1]
                
            if len(rejected_response_ids) > self.max_response_length - 1:
                rejected_response_ids = rejected_response_ids[:self.max_response_length - 1]
            
            #IGNORE_INDEX to ignore context_indices during loss computation.

            chosen_input_ids = chosen_context_ids + [self.tokenizer.bos_token_id] + chosen_response_ids
            chosen_labels    = [IGNORE_INDEX] * (len(chosen_context_ids) + 1) + chosen_response_ids

            rejected_input_ids = rejected_context_ids + [self.tokenizer.bos_token_id] + rejected_response_ids
            rejected_labels    = [IGNORE_INDEX] * (len(rejected_context_ids) + 1) + rejected_response_ids
            
            chosen_input_ids = self.truncate(chosen_input_ids)
            chosen_labels = self.truncate(chosen_labels)
            rejected_input_ids = self.truncate(rejected_input_ids)
            rejected_labels = self.truncate(rejected_labels)
            
            assert len(chosen_input_ids) == len(chosen_labels), f'Context Length: {len(chosen_input_ids)} Chosen Response Length: {len(chosen_labels)}'
            assert len(rejected_input_ids) == len(rejected_labels), f'Context Length: {len(rejected_input_ids)} Chosen Response Length: {len(rejected_labels)}'         
            
            result["chosen_input_ids"].append(torch.LongTensor(chosen_input_ids))
            result["chosen_labels"].append(torch.LongTensor(chosen_labels))
            result["rejected_input_ids"].append(torch.LongTensor(rejected_input_ids))
            result["rejected_labels"].append(torch.LongTensor(rejected_labels))
        
        result = {k:torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=self.tokenizer.pad_token_id) for k,v in result.items()}

        result["chosen_attention_mask"] = result["chosen_input_ids"].ne(self.tokenizer.pad_token_id).long()
        result["rejected_attention_mask"] = result["rejected_input_ids"].ne(self.tokenizer.pad_token_id).long()

        return result

if __name__ == "__main__":
    
    model_name = "facebook/opt-1.3b"
    batch_size=8
    
    dataset = AnthrophicRLHFDataset(
        "../../data/anthrophic_rlhf_dataset/helpful-online/hh_rlhf_train.jsonl",
        "anthrophic-helpful-online"
    )
    
    from torch.utils.data.dataloader import DataLoader
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=AnthrophicRLHFDataCollator(
            model_name=model_name
        )
    )
    
    for data in dataloader:
        for k,v in data.items():
            if torch.is_tensor(v):
                print(f'{k} {v.shape}')
        exit(1)