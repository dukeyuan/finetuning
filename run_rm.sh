CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --model_type bloom --model_name_or_path bloomz-560m \
    --train_file_dir ./data/reward --validation_file_dir ./data/reward \
    --do_train True --max_train_samples 1000  --per_device_train_batch_size 1 \
    --do_eval True --evaluation_strategy steps --eval_steps 50 --max_eval_samples 10 --per_device_eval_batch_size 1 \
    --use_peft True --seed 42 \
    --num_train_epochs 1 --learning_rate 2e-5 --warmup_ratio 0.05 --weight_decay 0.001 \
    --logging_strategy steps --logging_steps 10 \
    --save_strategy steps --save_steps 500 --save_total_limit 3 \
    --max_source_length 256 --max_target_length 256 \
    --output_dir ./output/outputs-rm-v1 --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
    --torch_dtype float32 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True
