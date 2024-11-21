export HF_AUTH_TOKEN="Your_API_Token"
python3 qlora.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_auth --output_dir "./finetuned_output" \
--logging_steps 50 --save_strategy steps \
--data_seed 42 \
--max_train_samples 150 \
--save_steps 250 \
--save_total_limit 2 \
--evaluation_strategy steps \
--eval_dataset_size 5 \
--max_eval_samples 250 \
--per_device_eval_batch_size 1 \
--max_new_tokens 512 \
--dataloader_num_workers 1 \
--group_by_length \
--logging_strategy steps \
--remove_unused_columns False \
--do_train \
--do_eval \
--lora_r 64 \
--lora_alpha 16 \
--lora_modules all \
--double_quant \
--quant_type nf4 \
--fp16 \
--bits 4 \
--warmup_ratio 0.03 \
--lr_scheduler_type constant \
--gradient_checkpointing \
--dataset #FILL IN# \
--source_max_len 512 \
--target_max_len 256 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--max_steps 250 \
--eval_steps 187 \
--learning_rate 0.0002 \
--adam_beta2 0.999 \
--max_grad_norm 0.3 \
--lora_dropout 0.1 \
--weight_decay 0.0 \
--seed 0