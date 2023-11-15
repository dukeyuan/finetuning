from lora_tools import *

def load_model(model_args, model_class):
    if model_args.model_name_or_path ==  None:
        raise ValueError(f"Error, model_name_or_path is None, must be loaded from a pre-trained model")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model

def preprocessing_datasets(data_args, raw_datasets, type, training_args, tokenizer):
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    full_max_length = max_source_length + max_target_length

    def preprocess_function(examples): # Preprocessing the datasets 
        sources = []
        targets = []
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input:
                instruction = instruction + '\n' + input
            source = PROMPT_TEMPLATE.format_map({'instruction': instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources, truncation=True, max_length=max_source_length)
        tokenized_targets = tokenizer(targets, add_special_tokens=False, truncation=True, max_length=max_target_length)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)
            # Padding labels to full max length for Seq2SeqCollator
            labels = torch.LongTensor([IGNORE_INDEX] * (full_max_length - len(t)) + t)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        results = {'input_ids': all_input_ids, 'labels': all_labels}
        return results

    return tokenizer_datasets(data_args, raw_datasets, type, training_args, tokenizer, preprocess_function)

def initialize_trainer(training_args, data_args, model, tokenizer, train_dataset, eval_dataset):
    full_max_length = data_args.max_source_length + data_args.max_target_length

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    return Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, padding="max_length", max_length=full_max_length))

def main(example_name, model_dict, task_type):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry(example_name, model_args, data_args)    
    logger.warning(f"Model args: {model_args}")
    logger.warning(f"Data args: {data_args}")
    logger.warning(f"Training args: {training_args}")
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    
    set_seed(training_args.seed)     # Set seed before initializing model.
    
    if not model_args.model_type:
        raise ValueError("Please specify a model_type, e.g. llama, chatglm, bloom, etc.")
    model_class, tokenizer_class = model_dict[model_args.model_type]

    tokenizer = load_tokenizer(model_args, tokenizer_class)
    model = load_model(model_args, model_class)   # Load model
    model = load_peft_model(model_class, model_args, training_args, model, task_type)

    raw_datasets = get_datasets(data_args, model_args)
    train_dataset, max_train_samples = preprocessing_datasets(data_args, raw_datasets, 'train', training_args, tokenizer)
    eval_dataset, max_eval_samples = preprocessing_datasets(data_args, raw_datasets, 'validation', training_args, tokenizer)
   
    trainer = initialize_trainer(training_args, data_args, model, tokenizer, train_dataset, eval_dataset)
    training(training_args, trainer, max_train_samples, model, tokenizer)
    evaluation(training_args, trainer, max_eval_samples)

if __name__ == "__main__":
    main("run_sft", MODEL_CLASSES,  TaskType.CAUSAL_LM)
