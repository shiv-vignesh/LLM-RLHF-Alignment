
import json
import torch
from tabulate import tabulate

from src.utils.data_preprocessing import AnthrophicRLHFDataset, AnthrophicRLHFDataCollator
from src.modelling import load_model_from_ckpt
from src.metrics import compute_generation_metrics

def health_check():
    """
    check a sample input to ensure model is not spitting gibberish.
    """
    
    model, device = load_model_from_ckpt(model_kwargs, ckpt_dir, is_trainable=False)    
    model.eval()
    
    collate_fn = AnthrophicRLHFDataCollator(model_kwargs["model_name"])

    # Prepare a simple prompt
    # prompt = "Explain why reinforcement learning is useful for aligning large language models."
    prompt = f"Human: How do I learn to be more appreciative? "
    
    inputs = collate_fn.preprocess_sample(prompt, device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=collate_fn.eos_token_id
        )
                
    text = collate_fn.decode_sample(outputs)
    print("\n Model output:\n", text)   

def main():
    
    model, device = load_model_from_ckpt(model_kwargs, ckpt_dir, is_trainable=False)    
    model.eval()
    
    dataset = AnthrophicRLHFDataset(
        dataset_kwargs["jsonl_path"],
        data_dir_type=dataset_kwargs["data_dir_type"],
        no_multi_turn=dataset_kwargs["no_multi_turn"]
    )
    
    collate_fn = AnthrophicRLHFDataCollator(
            model_kwargs["model_name"]
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=dataset_kwargs["batch_size"],
        collate_fn=collate_fn.preprocess_inference
    )
    
    results = []
    
    for batch_idx, data_items in enumerate(dataloader):
        data_items = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in data_items.items()
        }        
        
        with torch.no_grad():
            chosen_outputs = model.generate(
                input_ids=data_items["chosen_context_ids"],
                attention_mask=data_items["chosen_attention_mask"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=collate_fn.eos_token_id
            )
            
        decoded_chosen = collate_fn.decode_sample(chosen_outputs)
        decoded_chosen = [dialogue.strip().split("Assistant:")[-1] for dialogue in decoded_chosen]
                        
        batch_result = compute_generation_metrics(
            decoded_chosen,
            data_items["chosen_responses"],
            data_items["chosen_contexts"]
        )
        
        results.extend(batch_result)
        
        # Pretty-print table
        table = []
        for m in batch_result:
            table.append([
                m["bleu"], 
                m["rouge1"], 
                m["rougeL"]
            ])
        print(tabulate(
            table, 
            headers=["BLEU", "ROUGE-1", "ROUGE-L"],
            floatfmt=".4f"
        ))
        print(f"\nProcessed batch {batch_idx + 1}/{len(dataloader)}\n")

    # Save all metrics to JSON
    with open("inference_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Compute and print overall averages
    avg_bleu = sum(m["bleu"] for m in results) / len(results)
    avg_r1 = sum(m["rouge1"] for m in results) / len(results)
    avg_rL = sum(m["rougeL"] for m in results) / len(results)
    print("\n=== Overall Averages ===")
    print(tabulate(
        [[avg_bleu, avg_r1, avg_rL]],
        headers=["Avg BLEU", "Avg ROUGE-1", "Avg ROUGE-L"],
        floatfmt=".4f"
    ))        


if __name__ == "__main__":
    
    model_kwargs = {
        "model_name":"facebook/opt-1.3b",
        "device":"cuda",
        "peft_path":None,
        "model_ckpt":None
    }
    
    dataset_kwargs = {
        "data_dir_type":"anthrophic-helpful-online",
        "jsonl_path":"data/anthrophic_rlhf_dataset/helpful-online/hh_rlhf_train.jsonl",
        "batch_size":16,
        "no_multi_turn":True,
        
        # "jsonl_path":"data/anthrophic_rlhf_dataset/helpful-online/hh_rlhf_test.jsonl",
        # "batch_size":16,
        # "no_multi_turn":True
    }    
   
    ckpt_dir = "OPT-1.3B-helpful-online-qvk-tuned/checkpoint-3500"
    
    main()
    # health_check()