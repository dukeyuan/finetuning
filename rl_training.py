from lora_tools import *
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

def load_rl_tokenizer(model_args, tokenizer_class):
    tokenizer = load_tokenizer(model_args, tokenizer_class)

    tokenizer_kwargs = {"cache_dir": model_args.cache_dir, "use_fast": True, "trust_remote_code": model_args.trust_remote_code}
    reward_tokenizer = AutoTokenizer.from_pretrained(model_args.reward_model_name_or_path, **tokenizer_kwargs)
    return tokenizer, reward_tokenizer

def load_model(model_args, model_class,training_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules= None,
        inference_mode=False,
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
    )
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}

    print(model_args.device_map, 'model_args.device_map')
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        trust_remote_code=model_args.trust_remote_code,
        peft_config=peft_config if training_args.use_peft else None,
    )

    print_trainable_parameters(model)
    # # Load reward model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.reward_model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
    )
    reward_model.to(device)
    return model, reward_model

def get_reward_score(reward_model, reward_tokenizer, question, answer, device):
    """
    Get the reward score for a given question and answer pair.
    """
    inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
    score = reward_model(**inputs).logits[0].cpu().detach()
    return score

def preprocessing_datasets(data_args, raw_datasets, type, training_args, tokenizer):
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    full_max_length = max_source_length + max_target_length

    def preprocess_function(examples):
        results = {"query": [],"input_ids": []}
        for instruction, input in zip(examples['instruction'], examples['input']):
            if input:
                instruction = instruction + '\n' + input
            source = PROMPT_TEMPLATE.format_map({'instruction': instruction})
            tokenized_question = tokenizer(source, truncation=True, max_length=max_source_length, padding="max_length", return_tensors="pt")
            results["query"].append(source)
            results["input_ids"].append(tokenized_question["input_ids"])
        return results

    return tokenizer_datasets(data_args, raw_datasets, type, training_args, tokenizer, preprocess_function)

def initialize_trainer(training_args, data_args, model_args, model, tokenizer, train_dataset):
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    config = PPOConfig(
        steps=training_args.max_steps,
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        log_with=model_args.log_with,
        batch_size=model_args.batch_size,
        mini_batch_size=model_args.mini_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=model_args.early_stopping,
        target_kl=model_args.target_kl,
        seed=training_args.seed,
        init_kl_coef=model_args.init_kl_coef,
        adap_kl_ctrl=model_args.adap_kl_ctrl,
        accelerator_kwargs={"project_dir": training_args.output_dir},
    )
    
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )
    return trainer, config

def training(training_args, data_args, trainer, config, model, reward_model, tokenizer, reward_tokenizer):
     # These arguments are passed to the `generate` function of the PPOTrainer
    generation_kwargs = {
        "max_new_tokens": data_args.max_target_length,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        total_steps = config.total_ppo_epochs
        for step, batch in tqdm(enumerate(trainer.dataloader)):
            if step >= total_steps:
                break
            question_tensors = batch["input_ids"]
            question_tensors = [torch.LongTensor(i).to(device).squeeze(0) for i in question_tensors]
            responses = []
            response_tensors = []
            for q_tensor in question_tensors:
                response_tensor = trainer.generate(q_tensor, return_prompt=False, **generation_kwargs)
                r = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)[0]
                responses.append(r)
                response_tensors.append(response_tensor.squeeze(0))
            batch["response"] = responses

            # Compute reward score
            score_outputs = [
                get_reward_score(reward_model, reward_tokenizer, q, r, device) for q, r in
                zip(batch["query"], batch["response"])
            ]
            rewards = [torch.tensor(float(score) - training_args.reward_baseline) for score in score_outputs]
            
            # Run PPO step
            try:
                stats = trainer.step(question_tensors, response_tensors, rewards)
                trainer.log_stats(stats, batch, rewards)
                logger.debug(f"Step {step}/{total_steps}: reward score:{score_outputs}")
            except ValueError as e:
                logger.warning(f"Failed to log stats for step {step}, because of {e}")

            if step and step % training_args.save_steps == 0:
                save_dir = os.path.join(training_args.output_dir, f"checkpoint-{step}")
                save_model(save_dir)
        # Save final model
        save_model(trainer.model, tokenizer, training_args)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.warning(f"model args: {model_args}")
    logger.warning(f"data_args args: {data_args}")
    logger.warning(f"Parse args: {training_args}")
    set_seed(training_args.seed)     # Set seed before initializing model.

    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    tokenizer, reward_tokenizer = load_rl_tokenizer(model_args, tokenizer_class)
    model, reward_model = load_model(model_args, model_class, training_args)

    raw_datasets = get_datasets(data_args, model_args)
    train_dataset, max_train_samples = preprocessing_datasets(data_args, raw_datasets, 'train', training_args, tokenizer)

    trainer, config = initialize_trainer(training_args, data_args, model_args, model, tokenizer, train_dataset)
    training(training_args, data_args, trainer, config, model, reward_model, tokenizer, reward_tokenizer)

if __name__ == "__main__":
    main()
