import os

import torch
from transformers.trainer import Trainer
from transformers import TrainingArguments

from src.utils.data_preprocessing import AnthrophicRLHFDataCollator
from src.utils.utils import IGNORE_INDEX, PAD_TOKEN_ID

def create_training_arguments(trainer_kwargs):

    training_args = TrainingArguments(
        output_dir=trainer_kwargs["output_dir"],
        save_strategy=trainer_kwargs["save_strategy"],
        save_steps=trainer_kwargs["save_steps"],
        logging_steps=trainer_kwargs["logging_steps"],
        report_to=trainer_kwargs["report_to"],
        overwrite_output_dir=True
    )

    return training_args

class RewardModelTrainer(Trainer):
    
    def __init__(self, model_name:str, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, 
                model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), 
                optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
        
        self.metrics = dict()
        self.model_name = model_name
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
                
    
    def get_train_dataloader(self):

        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=AnthrophicRLHFDataCollator(
                model_name=self.model_name
            ),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers
        )
    
    def prediction_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        return super().prediction_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
    
    def get_eval_dataloader(self, eval_dataset = None):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=AnthrophicRLHFDataCollator(
                model_name=self.model_name
            ),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers
        )
    
    def get_test_dataloader(self, test_dataset):
        return super().get_test_dataloader(test_dataset)
    
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        
        #shape of value is (bs, max_seq_len, 1)
        
        _, chosen_clm_loss, chosen_clm_value = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"],
            return_dict=True
        ) 
        
        _, _, reject_clm_value = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"],
            return_dict=True
        )
        
        # create mask to ignore values from IGNORE_INDEX values
        chosen_action_mask = inputs["chosen_labels"].ne(IGNORE_INDEX).long()
        rejected_action_mask = inputs["rejected_labels"].ne(IGNORE_INDEX).long()
        
        chosen_clm_value = chosen_clm_value * chosen_action_mask
        reject_clm_value = reject_clm_value * rejected_action_mask
        
        #determine the seq length of each item of the batch
        bs = chosen_clm_value.shape[0]
        pad_token_id = self.get_train_dataloader().collate_fn.pad_token_id
        
        chosen_seq_len = (inputs["chosen_input_ids"].ne(pad_token_id).sum(-1) - 1).to(chosen_clm_value.device)
        reject_seq_len = (inputs["rejected_input_ids"].ne(pad_token_id).sum(-1) - 1).to(reject_clm_value.device)
        
        chosen_end_token_value = chosen_clm_value[torch.arange(bs, device=chosen_clm_value.device), chosen_seq_len]
        reject_end_token_value = reject_clm_value[torch.arange(bs, device=reject_clm_value.device), reject_seq_len]
        
        #use_last_token_value to compute reward
        loss1 = -torch.nn.functional.logsigmoid(chosen_end_token_value - reject_end_token_value).mean()        
        loss2 = chosen_clm_loss
        
        loss = loss1 + loss2
        
        outputs = dict(
            chosen_end_token_value=chosen_end_token_value,
            reject_end_token_value=reject_end_token_value,
            loss=loss
        )
        
        return (loss, outputs) if return_outputs else loss    

    def training_step(self, model, inputs, num_items_in_batch = None):
        return super().training_step(model, inputs, num_items_in_batch)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        
        for k, v in outputs.items():
            self.metrics[k] = v.mean()
            
        batch_idx = self.state.global_step % len(self.get_eval_dataloader())
        
        #TODO, fix data collateFn to return just prompts and not prompt + response ids.        
        # if batch_idx % 100 == 0:
        #     # Example: generate text from input_ids
        #     generated_ids = model.generate(
        #         inputs["chosen_input_ids"],
        #         attention_mask=inputs["chosen_attention_mask"],
        #         max_new_tokens=50
        #     )
        #     # Decode and compute generation metrics
        #     # Example using a simple placeholder function
        #     generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #     reference_texts = self.tokenizer.batch_decode(inputs["chosen_labels"], skip_special_tokens=True)
        #     # Compute your custom metric function
        #     gen_metrics = compute_generation_metrics(generated_texts, reference_texts)
        #     # Store generation metrics
        #     for k, v in gen_metrics.items():
        #         self.metrics[f"gen_{k}"] = v

        logits = tuple(v for k, v in outputs.items() if k in ["accepts_end_token_value", "rejects_end_token_value"])
        if prediction_loss_only:
            return (loss, None, None)
        
        logits = torch.stack(logits, dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device)
        return loss, logits, labels
    
    def log(self, logs, start_time = None):
        if len(self.metrics) > 0:
            for k, v in self.metrics.items():
                logs[f"eval_{k}"] = v.item()
            self.metrics.clear()

        return super().log(logs)

    def _save_checkpoint(self, model, trial):
        
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        os.makedirs(output_dir, exist_ok=True)  # ensures the directory exists
        
        torch.save(
            model.state_dict(), f'{output_dir}/ckpt-model.pt'
        )

        return super()._save_checkpoint(model, trial)
    
    def get_state_dict(self):
        pass
    
    def _save(self, output_dir = None, state_dict=None):
        """
        call get_state_dict()
        """
        pass
    

    