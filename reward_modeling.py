from lora_tools import *

def load_model(model_args, model_class):
    if model_args.model_name_or_path == None:
        raise ValueError(f"Error, model_name_or_path is None, must be loaded from a pre-trained model")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}
    if model_args.model_type in ['bloom', 'llama']:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            load_in_8bit=model_args.load_in_8bit,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
        )
        model.score = CastOutputToFloat(model.score)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True
        )
        model.to(training_args.device)
    
    return model

def preprocessing_datasets(data_args, raw_datasets, type, training_args, tokenizer):
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    full_max_length = max_source_length + max_target_length
    def preprocess_function(examples):
        """ Turn the dataset into pairs of Question + Answer, where input_ids_chosen is the preferred question + answer and text_rejected is the other. """
        new_examples = {"input_ids_chosen": [], "attention_mask_chosen": [], "input_ids_rejected": [], "attention_mask_rejected": []}
        for question, chosen, rejected in zip(examples["question"], examples["response_chosen"], examples["response_rejected"]):
            tokenized_chosen = tokenizer("Question: " + question + "\n\nAnswer: " + chosen, max_length=full_max_length)
            tokenized_rejected = tokenizer("Question: " + question + "\n\nAnswer: " + rejected, max_length=full_max_length)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

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
    return RewardTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=full_max_length, padding="max_length"),
    )

@dataclass
class RewardDataCollatorWithPadding:
    """We need to define a special data collator that batches the data in our chosen vs rejected format"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append({"input_ids": feature["input_ids_chosen"],"attention_mask": feature["attention_mask_chosen"]})
            features_rejected.append({"input_ids": feature["input_ids_rejected"],"attention_mask": feature["attention_mask_rejected"]})
        batch_chosen = self.tokenizer.pad(
            features_chosen,padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch

class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

def main(example_name, model_dict, task_type):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry(example_name, model_args, data_args)
    logger.add("./logs/logtest.log")
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
    # evaluation(training_args, trainer, max_eval_samples)
    
if __name__ == "__main__":
    main("run_rm", MODEL_REWARD_CLASSES, TaskType.SEQ_CLS)
