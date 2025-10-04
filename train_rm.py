from src.utils.data_preprocessing import AnthrophicRLHFDataset, AnthrophicRLHFDataCollator
from src.modelling import create_model
from src.trainer.train_reward_model import create_training_arguments, RewardModelTrainer
from src.metrics import compute_metrics_for_pair

def main():    
    model = create_model(model_kwargs, training_kwargs["lora_kwargs"])
    
    train_dataset = AnthrophicRLHFDataset(
        dataset_kwargs["train_dataset"]["jsonl_path"],
        dataset_kwargs["data_dir_type"],
        dataset_kwargs["train_dataset"]["no_multi_turn"],
    )
    
    if dataset_kwargs["eval_dataset"]:
        eval_dataset = AnthrophicRLHFDataset(
            dataset_kwargs["eval_dataset"]["jsonl_path"],
            dataset_kwargs["data_dir_type"],
            dataset_kwargs["eval_dataset"]["no_multi_turn"],
        )
    else:
        eval_dataset = None
        
    if dataset_kwargs["test_dataset"]:
        test_dataset = AnthrophicRLHFDataset(
            dataset_kwargs["test_dataset"]["jsonl_path"],
            dataset_kwargs["data_dir_type"],
            dataset_kwargs["test_dataset"]["no_multi_turn"],
        )
        
    else:
        test_dataset = None

    training_args = create_training_arguments(training_kwargs)
    
    trainer = RewardModelTrainer(
        model_name=model_kwargs["model_name"],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,        
        compute_metrics=compute_metrics_for_pair #TODO, add compute_metrics func
    )
    
    trainer.train()

if __name__ == "__main__":
    
    model_kwargs = {
        "model_name":"facebook/opt-1.3b",
        "device":"cuda",
        "peft_path":None,
        "model_ckpt":None
    }
    
    training_kwargs = {
        "lora_kwargs":{
            "inference_mode":False,
            "lora_target":["q_proj","v_proj","k_proj","o_proj"],
            "lora_rank":128,
            "lora_alpha":32,
            "lora_dropout":0.05,
        },
        "output_dir":"tmp_trainer",
        "save_strategy":"steps",
        "save_steps":500,
        "logging_steps":50,
        "report_to":"tensorboard"
    }

    dataset_kwargs = {
        "data_dir_type":"anthrophic-helpful-online",
        "train_dataset":{
            "jsonl_path":"data/anthrophic_rlhf_dataset/helpful-online/hh_rlhf_train.jsonl",
            "batch_size":8,
            "no_multi_turn":True
        },
        "eval_dataset":{},
        "test_dataset":{
            "jsonl_path":"data/anthrophic_rlhf_dataset/helpful-online/hh_rlhf_test.jsonl",
            "batch_size":16,
            "no_multi_turn":True
        }
    }
    
    main()