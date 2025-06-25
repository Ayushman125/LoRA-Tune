import torch
import os
import json # For handling custom JSON datasets
import matplotlib.pyplot as plt # For plotting
import glob # For finding checkpoint directories

# Hugging Face Transformers and PEFT for LLM Fine-tuning
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling # For handling padding and labels for CLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, Dataset # For handling datasets

# --- Configuration for Model, Quantization, and PEFT (Global Scope) ---
# These are defined globally so child processes can access them.
MODEL_ID = "distilbert/distilgpt2" # Example: A small, accessible model for testing
DATASET_ID = "Abirate/english_quotes" # A small, simple text dataset
DATASET_TEXT_COLUMN = "quote"
MAX_SEQUENCE_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. Training will be extremely slow on CPU.")
    print("Consider installing CUDA toolkit and a compatible PyTorch version if you have an NVIDIA GPU.")

# 4-bit Quantization Configuration (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Corrected from load_in_44bit, assuming typo
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
)

# LoRA (Low-Rank Adaptation) Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"], # Example for distilgpt2-like models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"Quantization compute dtype set to: {bnb_config.bnb_4bit_compute_dtype}")
print(f"LoRA Config: r={lora_config.r}, lora_alpha={lora_config.lora_alpha}, target_modules={lora_config.target_modules}")

print(f"Loading tokenizer for '{MODEL_ID}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Added '[PAD]' as pad_token to tokenizer.")
else:
    print(f"Tokenizer already has a pad_token: '{tokenizer.pad_token}'")

tokenizer.padding_side = "right"
print(f"Tokenizer padding_side set to: '{tokenizer.padding_side}'")

print(f"Loading model '{MODEL_ID}' with 4-bit quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Access vocab size via model.config.vocab_size
    if len(tokenizer) != model.config.vocab_size:
        old_vocab_size = model.config.vocab_size
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings from {old_vocab_size} to {len(tokenizer)} to match tokenizer vocab.")
    model.eval()

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the MODEL_ID is correct and you have network access.")
    if "HuggingFaceHub" in str(e) and "token" in str(e):
        print("You might need to provide a Hugging Face token for private models. Run: huggingface-cli login")
    raise

model = prepare_model_for_kbit_training(model)
print("Model prepared for k-bit training (gradient checkpointing enabled).")
model = get_peft_model(model, lora_config)
print("\nTrainable parameters after applying LoRA:")
model.print_trainable_parameters()

# --- Functions (Global Scope for Multiprocessing) ---
# These functions need to be defined in the global scope so child processes can pick them up.
# They will access the globally defined `tokenizer`.
def tokenize_function(examples):
    return tokenizer(
        examples[DATASET_TEXT_COLUMN],
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length"
    )

def prepare_clm_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

def plot_training_metrics(log_history):
    print("\nPlotting training metrics...")
    steps = []
    losses = []
    eval_steps = []
    eval_losses = []

    for log in log_history:
        if "loss" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])

    if not steps:
        print("No training steps logged for plotting training loss.")
        return

    num_subplots = 2 if eval_losses else 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(12, 5))

    if num_subplots == 1:
        axes = [axes]

    axes[0].plot(steps, losses, label='Training Loss', color='blue')
    axes[0].set_title('Training Loss Over Steps')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    if eval_losses and num_subplots > 1:
        axes[1].plot(eval_steps, eval_losses, label='Evaluation Loss', color='orange')
        axes[1].set_title('Evaluation Loss Over Steps')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
    elif eval_losses and num_subplots == 1:
        axes[0].plot(eval_steps, eval_losses, label='Evaluation Loss', color='orange', linestyle='--')
        axes[0].legend()

    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    # NEW: Flag to control training vs. loading for inference
    LOAD_MODEL_FOR_INFERENCE_ONLY = False # <--- SET THIS TO TRUE TO SKIP TRAINING AND LOAD MODEL

    # Paths for saved models/adapters
    output_adapter_dir = "./final_lora_adapters"
    merged_output_dir = "./merged_fine_tuned_model"
    training_output_dir = "./llm_finetuning_results" # Where Trainer saves its checkpoints

    # Define inference_model and inference_tokenizer here with a default None
    # so they are always in scope, regardless of the execution path.
    inference_model = None
    inference_tokenizer = None

    if not LOAD_MODEL_FOR_INFERENCE_ONLY:
        # Determine if resuming training is possible
        resume_from_checkpoint = None
        # Check if there are any checkpoints in the output directory
        checkpoints = list(glob.glob(os.path.join(training_output_dir, "checkpoint-*")))
        if len(checkpoints) > 0:
            # Sort to get the latest checkpoint if multiple exist
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            resume_from_checkpoint = checkpoints[-1]
            print(f"\nFound existing checkpoint: {resume_from_checkpoint}. Resuming training...")
        else:
            print("\nNo existing checkpoints found. Starting training from scratch.")

        # --- Data Loading and Preprocessing ---
        print(f"\nLoading dataset '{DATASET_ID}'...")

        def load_and_prepare_data():
            try:
                raw_datasets = load_dataset(DATASET_ID)
                print(f"Loaded dataset from Hugging Face Hub: {DATASET_ID}")
                print(raw_datasets)
                return raw_datasets

            except Exception as e:
                print(f"Could not load dataset '{DATASET_ID}' from Hugging Face Hub: {e}")
                print("Attempting to create a dummy dataset for demonstration.")
                dummy_data = {
                    "train": [
                        {DATASET_TEXT_COLUMN: "Hello, this is a sample sentence for fine-tuning."},
                        {DATASET_TEXT_COLUMN: "Another example text to train the language model."},
                        {DATASET_TEXT_COLUMN: "Large language models are powerful."},
                        {DATASET_TEXT_COLUMN: "Fine-tuning improves model performance on specific tasks."},
                        {DATASET_TEXT_COLUMN: "I enjoy learning about AI and machine learning."}
                    ],
                    "validation": [
                        {DATASET_TEXT_COLUMN: "This is a validation sentence."},
                        {DATASET_TEXT_COLUMN: "Checking the model's generalization."}
                    ]
                }
                train_ds = Dataset.from_list(dummy_data["train"])
                val_ds = Dataset.from_list(dummy_data["validation"])
                raw_datasets = {"train": train_ds, "validation": val_ds}
                print("Created a dummy dataset for demonstration purposes.")
                print(raw_datasets)
                return raw_datasets

        raw_datasets = load_and_prepare_data()

        print(f"Tokenizing dataset with max_length={MAX_SEQUENCE_LENGTH}...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count() // 2,
            remove_columns=raw_datasets["train"].column_names
        )

        tokenized_datasets = tokenized_datasets.map(prepare_clm_labels, batched=True, num_proc=os.cpu_count() // 2)

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else None
        if not eval_dataset and "test" in tokenized_datasets:
            eval_dataset = tokenized_datasets["test"]

        if not eval_dataset:
            print("No validation/test split found. Creating a small validation set from training data.")
            train_size = int(0.9 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])

        print(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            print(f"Validation dataset size: {len(eval_dataset)}")
        else:
            print("No separate validation set available.")

        # --- Training Arguments ---
        training_args = TrainingArguments(
            output_dir=training_output_dir, # Use the defined output dir
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=25,
            save_steps=100,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            fp16=False,
            bf16=True if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else False,
            report_to="tensorboard",
            dataloader_num_workers=os.cpu_count() // 2,
            push_to_hub=False,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )

        # --- Data Collator ---
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # --- Initialize Trainer ---
        print("\nInitializing Hugging Face Trainer...")
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        except Exception as e:
            print(f"Error initializing Trainer: {e}")
            print("Check your TrainingArguments and dataset preparation.")
            exit()

        # --- Train the Model ---
        print("\nStarting training...")
        try:
            # Pass the resume_from_checkpoint path here
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            print("Training complete!")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            print("Common issues: Out of memory (try reducing batch size, sequence length), corrupted data, incorrect target modules.")
            exit()

        # --- Assign trained model/tokenizer for inference below ---
        inference_model = model # This is the fine-tuned PEFT model
        inference_tokenizer = tokenizer

        # --- Save the fine-tuned LoRA adapters ---
        os.makedirs(output_adapter_dir, exist_ok=True)
        trainer.model.save_pretrained(output_adapter_dir)
        tokenizer.save_pretrained(output_adapter_dir)
        print(f"Fine-tuned LoRA adapters and tokenizer saved to: {output_adapter_dir}")

        # --- Optional: Merge LoRA Adapters with Base Model and Save (for deployable model) ---
        try:
            print("\nAttempting to merge LoRA adapters with base model for full model save...")
            base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                return_dict=True,
                torch_dtype=bnb_config.bnb_4bit_compute_dtype if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            if len(tokenizer) != base_model_for_merge.config.vocab_size:
                old_vocab_size_merge = base_model_for_merge.config.vocab_size
                base_model_for_merge.resize_token_embeddings(len(tokenizer))
                print(f"Resized base model embeddings for merge from {old_vocab_size_merge} to {len(tokenizer)}.")


            merged_model = PeftModel.from_pretrained(base_model_for_merge, output_adapter_dir)
            merged_model = merged_model.merge_and_unload()

            os.makedirs(merged_output_dir, exist_ok=True)
            merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_output_dir)
            print(f"Merged fine-tuned model saved to: {merged_output_dir}")

        except Exception as e:
            print(f"\nCould not merge LoRA adapters with base model (likely due to memory constraints or unsupported operation): {e}")
            print("You can still load the base model and LoRA adapters separately for inference.")
            print("Example for separate loading (assuming you saved adapters):")
            print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
            print("   from peft import PeftModel")
            print("   model = AutoModelForCausalLM.from_pretrained('your_base_model_id', torch_dtype=torch.bfloat16, device_map='auto')")
            print("   model = PeftModel.from_pretrained(model, './final_lora_adapters')")
            print("   tokenizer = AutoTokenizer.from_pretrained('./final_lora_adapters')")

    else: # If LOAD_MODEL_FOR_INFERENCE_ONLY is True (skip training)
        print("\nSkipping training. Loading saved model/LoRA adapters for inference.")
        
        if os.path.exists(merged_output_dir) and os.path.isdir(merged_output_dir):
            try:
                print(f"Attempting to load merged model from: {merged_output_dir}")
                inference_model = AutoModelForCausalLM.from_pretrained(
                    merged_output_dir,
                    torch_dtype=bnb_config.bnb_4bit_compute_dtype if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                inference_tokenizer = AutoTokenizer.from_pretrained(merged_output_dir, trust_remote_code=True)
                print("Successfully loaded merged model for inference.")
            except Exception as e:
                print(f"Could not load merged model from {merged_output_dir}: {e}")
                inference_model = None # Reset to try adapters if merged load fails

        if inference_model is None: # If merged model loading failed or not present, try loading base + adapters
            if os.path.exists(output_adapter_dir) and os.path.isdir(output_adapter_dir):
                try:
                    print(f"Attempting to load base model and LoRA adapters separately from: {output_adapter_dir}")
                    inference_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_ID,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    inference_tokenizer = AutoTokenizer.from_pretrained(output_adapter_dir, trust_remote_code=True)

                    if inference_tokenizer.pad_token is None:
                        inference_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    if len(inference_tokenizer) != inference_model.config.vocab_size:
                        old_vocab_size_inference = inference_model.config.vocab_size
                        inference_model.resize_token_embeddings(len(inference_tokenizer))
                        print(f"Resized inference base model embeddings from {old_vocab_size_inference} to {len(inference_tokenizer)} to match tokenizer vocab.")

                    inference_model = PeftModel.from_pretrained(inference_model, output_adapter_dir)
                    inference_model = inference_model.eval()
                    print("Successfully loaded base model and LoRA adapters for inference.")

                except Exception as e:
                    print(f"Could not load base model and LoRA adapters from {output_adapter_dir}: {e}")
                    print("Please ensure the adapter directory exists and contains valid PEFT model files.")
                    exit()
            else:
                print(f"Neither merged model nor LoRA adapters found at {merged_output_dir} or {output_adapter_dir}.")
                print("Please run the training script first or ensure paths are correct.")
                exit()
        
        if inference_model is not None:
             inference_model.to(DEVICE)
        else:
             print("No model could be loaded. Exiting.")
             exit()

    # --- Inference Example ---
    print("\n--- Running a simple inference example ---")
    try:
        if inference_model is None or inference_tokenizer is None:
            print("Model or tokenizer not loaded. Cannot run inference.")
            exit()

        prompt_template = "The most important thing in life is "
        if DATASET_TEXT_COLUMN == "quote":
            prompt_template = "The best quote about wisdom is: "
        
        prompt = prompt_template

        print(f"\nPrompt: '{prompt}'")
        inputs = inference_tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_tokens = inference_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=inference_tokenizer.pad_token_id
            )

        generated_text = inference_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"\nGenerated Text:\n{generated_text}")

    except Exception as e:
        print(f"\nCould not run inference example: {e}")
        print("Ensure you have enough memory to load the model for inference.")
        print("If memory is an issue, try running generation on CPU or with smaller models.")

    # Only plot if training happened and trainer object exists
    if not LOAD_MODEL_FOR_INFERENCE_ONLY and 'trainer' in locals():
        plot_training_metrics(trainer.state.log_history)