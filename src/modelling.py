import os
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import AutoModelForCausalLMWithValueHead

from src.utils.utils import MODEL_CLASSES

def create_model(model_kwargs:dict, training_kwargs:dict):
    
    model_name = model_kwargs["model_name"]
    if model_name in MODEL_CLASSES:
        config, _, model_class = MODEL_CLASSES[model_name]
    else:
        config, _, model_class = MODEL_CLASSES["auto"]

    model = model_class.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=training_kwargs["inference_mode"],
        target_modules=training_kwargs["lora_target"],
        r=training_kwargs["lora_rank"],
        lora_alpha=training_kwargs["lora_alpha"],
        lora_dropout=training_kwargs["lora_dropout"]
    )
    
    model = get_peft_model(model, peft_config=lora_config)
        
    if torch.cuda.is_available() and model_kwargs["device"] == "cuda":
        device = torch.device(model_kwargs["device"])
    elif model_kwargs["device"]:
        device = torch.device(model_kwargs["device"])
    else:
        device = torch.device("cpu")

    model.to(device)

    if model_kwargs["peft_path"] is not None and os.path.exists(model_kwargs["peft_path"]):
        model = PeftModel.from_pretrained(model, model_kwargs["peft_path"])
        
    elif model_kwargs["model_ckpt"] is not None and os.path.exists(model_kwargs["model_ckpt"]):
        model.load_state_dict(torch.load(model_kwargs["model_ckpt"], map_location=device))
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model.config.use_cache = False
    
    #TODO, load v_head state_dict from peft_path
    #TODO, enable gradient ckpting
    
    return model