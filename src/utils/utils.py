import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

PROMPT_TEMPLATE = dict(
    llama_alpaca=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    llama2_alpaca=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    ),
    default=(
        "Human: {instruction}\nAssistant: "
    )
)

IGNORE_INDEX = -100
PAD_TOKEN_ID = None
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}