import os
import torch
from transformers import AutoModelForCausalLM
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

    if model_kwargs["peft_path"] is not None and os.path.exists(model_kwargs["peft_path"]):
        model = PeftModel.from_pretrained(model, model_kwargs["peft_path"])
        
    elif model_kwargs["model_ckpt"] is not None and os.path.exists(model_kwargs["model_ckpt"]):
        model.load_state_dict(torch.load(model_kwargs["model_ckpt"], map_location=device))
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model.config.use_cache = False
    
    model.to(device)
    
    #TODO, load v_head state_dict from peft_path
    #TODO, enable gradient ckpting
    
    return model, device

def load_model_from_ckpt(model_kwargs: dict, ckpt_dir: str, is_trainable=True):
    """
    Loads a PEFT (LoRA) + ValueHead model from a saved checkpoint folder
    like tmp_trainer-2/checkpoint-10/
    """
    base_model_name = model_kwargs["model_name"]
    
    if torch.cuda.is_available() and model_kwargs["device"] == "cuda":
        device = torch.device(model_kwargs["device"])
        device_map = "auto"
    elif model_kwargs["device"]:
        device = torch.device(model_kwargs["device"])
        device_map = "cpu"
    else:
        device = torch.device("cpu")
        device_map = "cpu"
    
    # Step 1: Load base LM
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, 
                                                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                                    device_map=device_map)

    # Step 2: Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, 
                                    ckpt_dir, 
                                    is_trainable=is_trainable, 
                                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                    device_map=device_map)

    # Step 3: Wrap with TRL value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model.to(device)

    # Step 4: Load v_head if saved inside adapter_model.bin
    adapter_path = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(adapter_path):
        state_dict = torch.load(adapter_path, map_location="cpu")
   
        v_head_state = {k.replace("v_head.", ""): v for k, v in state_dict.items() if k.startswith("v_head.")}
        if v_head_state:
            print(f"Loading v_head weights from {adapter_path}")
            model.v_head.load_state_dict(v_head_state, strict=False)
        else:
            print("⚠️ No v_head weights found in adapter_model.bin (might have been saved separately)")
    else:
        print(f"⚠️ No adapter_model.bin found in {ckpt_dir}")

    model.config.use_cache = False
    model.to(device)

    return model, device

def check_trainable_parameters(model:torch.nn.Module):

    trainable_params, total_params = 0, 0
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    print(f"Trainable params: {trainable_params:,d} | Total params: {total_params:,d}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")
    