
import os

HF_HUB_PATH="/mnt/tempten/huggingface/hub"

os.environ["HF_DATASETS_CACHE"] = HF_HUB_PATH
os.environ["HF_MODELS_CACHE"] =  HF_HUB_PATH
os.environ["TRANSFORMERS_CACHE"] = HF_HUB_PATH
os.environ["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = HF_HUB_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_PATH
os.environ['HF_HOME'] = HF_HUB_PATH

# Import the os module for interacting with the operating system (e.g., file paths, environment variables)
import os

# Import the load_dataset function from the datasets library to easily load and use datasets
from datasets import load_dataset

# Import the torch library for tensor computations and deep learning operations
import torch

# Import Qwen2VLForConditionalGeneration and Qwen2VLProcessor from transformers for model loading and preprocessing
from transformers import  BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

# Import LoraConfig and get_peft_model from peft for configuring and applying Low-Rank Adaptation (LoRA) to models
from peft import LoraConfig, get_peft_model
# LoraConfig: Used to define the configuration for LoRA, such as rank, alpha, and target modules.
# get_peft_model: Applies the LoRA configuration to a base model, returning a PEFT (Parameter-Efficient Fine-Tuning) model.

# Import SFTTrainer and SFTConfig from trl for supervised fine-tuning (SFT) of transformer models
from trl import SFTTrainer, SFTConfig
# SFTConfig: Used to specify training parameters for supervised fine-tuning (e.g., learning rate, batch size).
# SFTTrainer: Handles the training loop for supervised fine-tuning using the provided model, dataset, and configuration.

import warnings
warnings.filterwarnings("ignore")
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct" ## switching to 2.5 model
EPOCHS = 1
BATCH_SIZE = 8
GRADIENT_CHECKPOINTING = True,  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False,
OPTIM = "paged_adamw_32bit"
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_STEPS = 0
DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs
REMOVE_UNUSED_COLUMNS = False # VLM thing
MAX_SEQ_LEN=1024  # Increased for vision-language models


# we need to be able to fully format our database to usable prompts for training and testing . we use data formattter for that pupose that turns each data sample to role based prompts as openai standards
system_message = """You are an expert embryologist AI specialized in morphological assessment of human embryos for IVF applications.

CORE FUNCTIONS:
- Identify morphokinetic stages (tPNa, t2, t3, t4, t5, t6, t7, t8, t9+, tEB, tB, tHB, etc.)
- Apply Gardner grading system (AA, AB, BA, BB, BC, CB, CC, and degenerative grades with X)
- Assess embryo viability and transfer potential based on morphological features
- Detect developmental abnormalities, fragmentation, and quality indicators

ASSESSMENT APPROACH:
- Analyze only visible morphological characteristics
- Use standard embryological terminology and grading systems
- Provide objective evaluations without referencing transfer outcomes
- Include both stage codes (t2, t3, etc) and Gardner grades (AA, BB, etc) in responses
- Focus on cellular structure, symmetry, expansion, and developmental timing

Evaluate each embryo systematically using established clinical criteria to support embryo selection decisions.
Always base your assessment on what you can actually see in the image, not on assumptions."""

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]
## get 1% data

# train test split

dataset = load_dataset("csv", data_files=["/mnt/tempten/vlm/enhanced_embryo_vlm_data.csv"])
# Only keep the random 10000 examples
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(0, 50000))

# Split using HuggingFace datasets
# This creates a 60-20-20 split
split_dataset = dataset["train"].train_test_split(
    test_size=0.4,  # 40% for eval+test
    seed=42
)

# Split the test portion into eval and test
eval_test_split = split_dataset["test"].train_test_split(
    test_size=0.5,  # 50% of 40% = 20% each for eval and test
    seed=42
)

# Assign splits
train_dataset = split_dataset["train"]
eval_dataset = eval_test_split["train"]  # This is actually eval
test_dataset = eval_test_split["test"]

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Skip vision layers from quantization
        # llm_int8_skip_modules=["visual", "vision_model", "patch_embed", "blocks", "merger"]
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        quantization_config=bnb_config,
        use_cache=False
        )

else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        use_cache=False
        )
    
# ENHANCED: Ensure fast processing is enabled consistently
processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL_ID,
    use_fast=True,  # Enable fast tokenization
    trust_remote_code=True  # May be needed for some models
)
processor.tokenizer.padding_side = "right"


# ENHANCED: Verify fast tokenizer is being used
print(f"Using fast tokenizer: {processor.tokenizer.is_fast}")
if hasattr(processor.tokenizer, 'backend_tokenizer'):
    print(f"Tokenizer backend: {type(processor.tokenizer.backend_tokenizer)}")

def text_generator(sample_data):
    # Use the processor's chat template to convert the first two roles (system and user) into a prompt string.
    # tokenize=False means it returns a string, not token IDs.
    # add_generation_prompt=True appends the assistant's prompt for generation.
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )

    print(f"Prompt: {text}")
    print("-"*30)

    # Extract and preprocess the image from the user message
    image_path = sample_data[1]["content"][0]["image"]
    image_inputs = Image.open(image_path) if isinstance(image_path, str) else image_path
    
    # ENHANCED: Optimize image for inference
    if image_inputs.mode != 'RGB':
        image_inputs = image_inputs.convert('RGB')
    
    # Resize for consistent processing (optional for inference, but recommended)
    target_size = (448, 448)
    # image_inputs = image_inputs.resize(target_size, Image.Resampling.LANCZOS)
    
    # ENHANCED: Use fast processing for inference - NO truncation for VLMs
    inputs = processor(
        text=[text],            # List of prompt strings
        images=image_inputs,    # Image or list of images
        return_tensors="pt",    # Return PyTorch tensors
        padding=True            # Enable padding for batch processing
        # Note: No truncation for VLMs to preserve image tokens
    )
    # Move the input tensors to the correct device (CPU or GPU).
    inputs = inputs.to(device)

    # Generate output token IDs from the model, limiting to MAX_SEQ_LEN new tokens.
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

    # Decode the generated token IDs back to text, skipping special tokens.
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    # Free up memory by deleting the input tensors.
    del inputs

    # Extract the actual answer from the sample data for comparison.
    actual_answer = sample_data[2]["content"][0]["text"]
    return output_text[0], actual_answer, image_inputs

# Generate text and compare with the actual answer.

# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     target_modules=["q_proj", "v_proj"],
#     task_type="CAUSAL_LM",
# )
peft_config=LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=[
        # Attention modules (core for vision-language understanding)
        "q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    # Only adapt the last few layers (most task-specific)
    layers_to_transform=list(range(24, 28)),  # Last 4 layers for 7B model
)

print(f"Before adapter parameters: {model.num_parameters()}")
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters() # After LoRA trainable parameters increases. Since we add adapter.

def collate_fn(examples):
    #  Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize just the assistant response to get its length
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels

    ####
    # Only unmask assistant response tokens
    for i, example in enumerate(examples):
        # Get the assistant response
        assistant_response = example[2]["content"][0]["text"]
        
        # Tokenize just the assistant response to get its length
        assistant_tokens = processor.tokenizer.encode(assistant_response, add_special_tokens=False)
        
        # Find where assistant response starts in the full sequence
        # This is approximate - you may need to adjust based on your chat template
        sequence_length = batch["input_ids"][i].shape[0]
        assistant_start = sequence_length - len(assistant_tokens)
        
        # Unmask only the assistant tokens
        labels[i, assistant_start:sequence_length] = batch["input_ids"][i, assistant_start:sequence_length]
    
    
    
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
    # Still mask padding and image tokens
  
    




    # Ignore the image token index in the loss computation (model specific)
  
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

   

output_dir="/mnt/tempten/vlm/output"

# ENHANCED: Add checkpoint resuming functionality
def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# Check for existing checkpoints
resume_from_checkpoint = get_latest_checkpoint(output_dir)
if resume_from_checkpoint:
    print(f"Resuming training from: {resume_from_checkpoint}")
else:
    print("Starting training from scratch")

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    dataset_kwargs=DATASET_KWARGS,
    max_seq_length=MAX_SEQ_LEN,
    remove_unused_columns = REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
    # ENHANCED: Additional performance optimizations
    dataloader_num_workers=4,  # Use multiple workers for data loading
    dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
    fp16=True if device == "cuda" else False,  # Use mixed precision on GPU
    # ENHANCED: Checkpoint resuming settings
    resume_from_checkpoint=resume_from_checkpoint,  # Resume from latest checkpoint
    save_total_limit=3,  # Keep only last 3 checkpoints to save disk space
    report_to=None,  # Disable wandb/tensorboard if not needed
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
)

print("Training")
# ENHANCED: The trainer will automatically resume from checkpoint if specified
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
print("-"*30)





trainer.save_model(training_args.output_dir)

metric = trainer.evaluate()
print(metric)
print("-"*30)
