# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
### Summary of Key Terms and Concepts

This script is designed for fine-tuning large language models (LLMs) using methods that optimize for memory and computational efficiency. Below are explanations of key terms used throughout the code:

1. **Fine-Tuning**:
   - Fine-tuning is the process of adapting a pre-trained model to a specific task or dataset by training it further on task-specific data. This typically involves modifying the model's weights using gradient descent.

2. **Parameter-Efficient Fine-Tuning (PEFT)**:
   - PEFT is a technique that enables fine-tuning of large models while modifying only a small subset of parameters, significantly reducing memory and computational requirements. This is achieved by freezing most of the model's parameters and introducing trainable components (e.g., adapters).

3. **Low-Rank Adaptation (LoRA)**:
   - LoRA is a PEFT method that injects small trainable matrices into the model's architecture. These matrices adapt the pre-trained model for specific tasks by learning low-rank updates to the original model's weights. This approach allows fine-tuning with minimal additional parameters.

4. **Adapters**:
   - Adapters are additional layers or modules inserted into a pre-trained model to enable task-specific fine-tuning. In PEFT methods like LoRA, adapters are small, efficient, and designed to avoid modifying the original model's weights.

5. **Quantization**:
   - Quantization reduces the precision of model parameters (e.g., from 32-bit floating point to 4-bit integers) to decrease memory usage and improve inference speed. This script uses quantization configurations to train and deploy large models efficiently.

6. **Gradient Checkpointing**:
   - A memory optimization technique where intermediate activations during forward passes are recomputed during the backward pass instead of being stored. This reduces memory usage but increases computation time.

7. **BitsAndBytes (bnb)**:
   - A library that provides support for low-bitwidth optimizations (e.g., 4-bit or 8-bit quantization) for PyTorch models. It enables efficient training and inference of large-scale models.

8. **Hugging Face Transformers**:
   - A library for state-of-the-art natural language processing (NLP) with pre-trained models like GPT, BERT, and more. It provides tools for tokenization, model loading, fine-tuning, and text generation.

Understanding these terms can help users/readers with the purpose and functionality of the script, which is to fine-tune large models efficiently using techniques like LoRA and PEFT while minimizing resource consumption.
"""

# Importing necessary libraries for training, evaluation, and dataset handling
from collections import defaultdict   # For grouping multiple values per key
import copy      # To create deep copies of objects
import json      # For JSON file handling
import os        # For interacting with the operating system
from os.path import exists, join, isdir    # Path utilities
from dataclasses import dataclass, field   # To define structured data classes
import sys      # For system-specific parameters and functions
from typing import Optional, Dict, Sequence   #For type annotations

# Third-party library imports
import numpy as np       # For numerical computations
from tqdm import tqdm       # For progress visualization
import logging              # For tracking and debugging script execution
import bitsandbytes as bnb   # Library for efficient memory usage
import pandas as pd           #For data manipulation and analysis
import importlib                # To dynamically import modules
from packaging import version         # For handling version comparisons
from packaging.version import parse   # To parse version strings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support    # For metric evaluation

# PyTorch and Hugging Face imports
import torch
import transformers     # Hugging Face Transformers library for NLP tasks
from torch.nn.utils.rnn import pad_sequence  # For padding sequences to uniform length
import argparse              # For parsing command-line arguments
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset    # Hugging Face datasets library
import evaluate

from peft import (        # Parameter-efficient fine-tuning library
    prepare_model_for_kbit_training,   # Prepares model for low-bit fine-tuning
    LoraConfig,      # Configuration for LoRA tuning
    get_peft_model,  # Initializes PEFT model
    PeftModel        # Base class for PEFT models
)
from peft.tuners.lora import LoraLayer    # Fine-tuning module for LoRA
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR   # Utility for checkpoint management

# Mock data for initial metric testing during development
# Ensures that the precision-recall metric pipeline is functional before applying to real data
labels = ["hello world", "goodbye world"]
predictions = ["hello world", "bad prediction"]

# Test metrics computation for mock data
precision, recall, f1, _ = precision_recall_fscore_support(
    labels, predictions, average='weighted', zero_division=0
)
print("Mock Precision:", precision, "Mock Recall:", recall, "Mock F1:", f1)

# Environment validation: Ensure the Hugging Face token is available
hf_auth_token = os.environ.get("HF_AUTH_TOKEN")
if not hf_auth_token:
    raise ValueError("Hugging Face token not found. Please set the 'HF_AUTH_TOKEN' environment variable.")

def is_ipex_available():
    """
    Check if Intel Extension for PyTorch (IPEX) is installed and compatible with the current PyTorch version.
    Ensures reproducibility by validating the environment setup.
    """
    def get_major_and_minor_from_version(full_version):
        """
        Extracts the major and minor version from a full version string.

        Args:
            full_version (str): Full version string (e.g., '1.12.1').

        Returns:
            str: Major and minor version as a string (e.g., '1.12').
        """
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        # Attempt to retrieve the IPEX version
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    # Compare major and minor versions of PyTorch and IPEX
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    
# Enable TensorFloat32 mode for faster computations on compatible GPUs
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

# Setting up the logger
logger = logging.getLogger(__name__)

# Constants for handling ignored indices in labels and default padding token
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

# Model configuration arguments with additional annotations for reproducibility
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
# Dataset-related configurations, ensuring flexibility and reproducibility
@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        # changed by author of the research paper
        # default=1024, metadata={"help": "Size of validation dataset."}
        default=1, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

# Training arguments with optimizations for efficient training
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    # Directory to cache model files
    cache_dir: Optional[str] = field(
        default=None
    )
    # Option to train on source text along with target text
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    # Configuration for MMLU evaluation dataset
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )

    # Full model fine-tuning (without adapters)
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    # Optimizer configurations
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    # For parameter-efficient fine-tuning
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    # Memory and logging settings
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
        
    # Optimizer configuration for low-memory training with LoRA
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # using lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    # Learning rate scheduler and warmup configuration
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
     # Logging and saving configurations
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})


# Generation-specific arguments with support for diverse sampling methods
@dataclass
class GenerationArguments:
    """
    Arguments for configuring text generation during evaluation and prediction.
    """
    # Added support for hyperparameters specific to generation
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    """
    Find all linear layers in the model to apply LoRA.
    """
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    """
    Callback for saving PEFT (Parameter-Efficient Fine-Tuning) adapter checkpoints during training.
    """
    def save_model(self, args, state, kwargs):
        """
        Saves the PEFT adapter model and removes unnecessary files.
        """
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        # Save the adapter model
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # Remove full model file to save space

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        """
        Triggered when a checkpoint is saved during training.
        """
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """
        Finalizes training and saves the last checkpoint.
        """
        def touch(fname, times=None):
            """
            Creates or updates a completion marker file.
            """
            with open(fname, 'a'):
                os.utime(fname, times)

                # Mark training completion

        touch(join(args.output_dir, 'completed'))
                # Save the final model

        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):
    """
    Load the model with memory-efficient settings and LoRA adapters.
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # Handle distributed settings
    if os.environ.get('LOCAL_RANK') is not None:
        """
    Checks if the model is running in a distributed training environment. 
    If yes, assigns a specific GPU/device (local_rank) for processing.
    """
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if args.full_finetune:
        """
    Ensures that only valid bit precision settings (16 or 32) are used 
    when fully fine-tuning the model.
    """
        assert args.bits in [16, 32]

    # Load the base model
    print(f'Loading base model {args.model_name_or_path}...')
    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )

    # Ensure special tokens are set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': "[EOS]"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': "[BOS]"})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': "[UNK]"})

    # Resize token embeddings if new tokens are added
    model.resize_token_embeddings(len(tokenizer))
    """
Updates the model's token embeddings to match the tokenizer's vocabulary size.
Necessary when adding new tokens or modifying the tokenizer.
"""

    if not args.full_finetune:
        """
    Prepares the model for parameter-efficient fine-tuning (k-bit training)
    using methods like LoRA or other adapters. Gradient checkpointing is enabled
    for efficient memory usage during training.
    """
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            """
        If a checkpoint directory is specified, loads the adapters from the checkpoint
        to continue training from the saved state.
        """
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            """
        If no checkpoint is provided, initializes and attaches LoRA modules
        (Low-Rank Adaptation) for fine-tuning.
        """
            print(f'Adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",             
            )
            model = get_peft_model(model, config)    # Applies the LoRA configuration

    # Adjust specific modules to the correct data types for efficient computation
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:      # Convert LoRA layers to bfloat16 if specified
                module = module.to(torch.bfloat16)
        if 'norm' in name:     # Ensure normalization layers use float32 for stability
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)    # Convert to bfloat16 for efficiency

    # Return the prepared model and tokenizer
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Adjusts the tokenizer and model embedding layers to accommodate new special tokens.
    """
    # Add new special tokens to the tokenizer
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # Resize the model's token embedding layer to match the updated tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        # Retrieve existing embedding weights
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data
        # Compute the average of existing embeddings to initialize new tokens
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        # Assign averaged embeddings to new tokens
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    """
    Data collator for causal language modeling tasks, preparing input and labels.
    """
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of instances into tokenized inputs and labels.
        """
        # Prepare sources and targets with special tokens
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize sources and targets
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Initialize input IDs and labels
        input_ids = []
        labels = []
        # Construct input and label sequences
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Pad sequences for consistent batch sizes
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        # Create the data dictionary
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    """
    Extracts input and output data from examples of unnatural instructions.
    Optionally includes reformulations.
    """
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    """
    Converts Alpaca dataset examples into the required input-output format based on their structure.
    """
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    """
    Loads a dataset based on its file extension and converts it into a usable format.
    Supported formats: JSONL, JSON, CSV, TSV.
    """
    if dataset_name.endswith('.jsonl'):
        if dataset_name.endswith('.jsonl'):
            full_dataset = Dataset.from_json(dataset_name)
            # Transform the dataset to extract `input` and `output`
            def transform_example(example):
                messages = example.get("messages", [])
                user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                assistant_message = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
                return {"input": user_message, "output": assistant_message}

        # Apply transformation
        full_dataset = full_dataset.map(transform_example, remove_columns=["messages"])
        return full_dataset
    elif dataset_name.endswith('.json'):
        full_dataset = Dataset.from_json(dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")
    # Split dataset into training and testing subsets
    split_dataset = full_dataset.train_test_split(test_size=2)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Prepares the dataset and data collator for fine-tuning.
    Converts raw data into tokenized and batched formats.
    """

    def load_data(dataset_path):
        """
        Loads and processes the dataset based on file type.
        """
        if dataset_path.endswith('.jsonl'):
            # Load JSON Lines dataset and transform to `input` and `output`
            full_dataset = Dataset.from_json(dataset_path)
            
            # Transform the dataset to extract `input` and `output`
            def transform_example(example):
                messages = example.get("messages", [])
                user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                assistant_message = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
                return {"input": user_message, "output": assistant_message}
            
            return full_dataset.map(transform_example, remove_columns=["messages"])
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

    # Load the dataset
    dataset = load_data(args.dataset)

    # Split the dataset into train and validation sets
    print('Splitting dataset into train and validation...')
    dataset = dataset.train_test_split(test_size=args.eval_dataset_size, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Debugging: Inspect raw data after splitting
    print("Raw Sample Input:", train_dataset[0]['input'])
    print("Raw Sample Output:", train_dataset[0]['output'])

    # Reduce dataset size for debugging or faster processing
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    # Tokenize the datasets
    def tokenize_data(dataset):
        def preprocess(example):
            tokenized_input = tokenizer(
                example['input'], truncation=True, max_length=args.source_max_len, padding=True
            )
            tokenized_output = tokenizer(
                example['output'], truncation=True, max_length=args.target_max_len, padding=True
            )
            return {
                "input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": tokenized_output["input_ids"],
            }

        return dataset.map(preprocess, batched=True)

    train_dataset = tokenize_data(train_dataset)
    eval_dataset = tokenize_data(eval_dataset)

    # Debugging: Inspect tokenized data
    print("Tokenized Input IDs:", train_dataset[0]['input_ids'])
    print("Tokenized Input Decoded:", tokenizer.decode(train_dataset[0]['input_ids']))
    print("Tokenized Output IDs:", train_dataset[0]['labels'])
    print("Tokenized Output Decoded:", tokenizer.decode(train_dataset[0]['labels']))

    # Define the data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

def get_last_checkpoint(checkpoint_dir):
    """
    Determines the last checkpoint in a given directory.
    - Returns None and True if training is already completed.
    - Otherwise, identifies and returns the last checkpoint path.
    """
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def normalize_output(text):
    """
    Cleans up text by removing padding, unwanted characters, and extra whitespace.
    """
    text = text.strip()
    text = text.replace("[PAD]", "").replace("\n", "").replace("[UNK]", "")
    text = " ".join(text.split())  # Normalize whitespace
    return text

def remove_padding(tokens, tokenizer):
    """
    Removes padding and special tokens from a list or tensor of token IDs.
    """
    # Flatten the tokens if it's a tensor or array
    if isinstance(tokens, (np.ndarray, torch.Tensor)):
        tokens = tokens.flatten().tolist()
    return [token for token in tokens if token not in [tokenizer.pad_token_id, IGNORE_INDEX]]


def post_process_prediction(pred):
    """
    Cleans predictions by removing special tokens and normalizing whitespace.
    """
    pred = pred.replace("<s>", "").replace("</s>", "").replace("[PAD]", "").strip()
    pred = " ".join(pred.split())
    return pred

def compute_metrics(eval_pred, tokenizer):
    """
    Computes precision, recall, and F1 score for predictions and labels.
    - Predictions and labels are decoded, normalized, and compared.
    """
    logits, labels = eval_pred

    # Convert logits to predictions
    predictions = np.argmax(logits, axis=-1)

    # Replace ignored indices in labels with padding token ID
    labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)

    # Flatten and remove padding for predictions and labels
    predictions = [remove_padding(pred.flatten(), tokenizer) for pred in predictions]
    labels = [remove_padding(label.flatten(), tokenizer) for label in labels]

    # Log raw predictions and labels
    print("Raw Predictions (Token IDs):", predictions[:5])
    print("Raw Labels (Token IDs):", labels[:5])

    # Decode and normalize predictions and labels
    decoded_predictions = [
        normalize_output(post_process_prediction(tokenizer.decode(pred, skip_special_tokens=True)))
        for pred in predictions
    ]
    decoded_labels = [
        normalize_output(post_process_prediction(tokenizer.decode(label, skip_special_tokens=True)))
        for label in labels
    ]

    # Debugging: Log decoded examples
    print("Sample Predictions (Decoded):", decoded_predictions[:5])
    print("Sample Labels (Decoded):", decoded_labels[:5])

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        decoded_labels, decoded_predictions, average='weighted', zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def safe_decode(token_ids, tokenizer):
    """
    Safely decodes token IDs into text, removing invalid tokens.
    """
    token_ids = [token for token in token_ids if token < len(tokenizer.get_vocab())]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def train():
    """
    Main training function:
    - Parses arguments.
    - Loads model and tokenizer.
    - Manages training, evaluation, and prediction loops.
    """
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    # Example of model inference
    test_input = tokenizer("Test input example", return_tensors="pt", padding=True, truncation=True)
    test_input = {key: val.to(model.device) for key, val in test_input.items()}
    outputs = model.generate(**test_input, max_new_tokens=50)
    print("Generated Prediction:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Adding callbacks and optional evaluation mechanisms
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        # Handle MMLU evaluation logic
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
        # Load and preprocess MMLU dataset based on specified split and sample limit
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        # Define token indices for options A, B, C, D for MMLU evaluation
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        # Load the accuracy evaluation metric
        accuracy = evaluate.load("accuracy")
        # Define MMLU evaluation callback for logging results
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                """
        Custom evaluation logic for MMLU.
        Processes logits, computes predictions, and calculates accuracy for each subject.
        """
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                # Iterate through batches and compute predictions and references
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                # Compute accuracy for each subject and overall
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                # Log results and restore original source length
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    # Initialize metrics dictionary for tracking results
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    # Modify evaluation/prediction decoding logic in `trainer.predict`
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        prediction_output = trainer.predict(trainer.eval_dataset)
        predictions = prediction_output.predictions
        labels = prediction_output.label_ids

        # Step 1: Convert logits to token IDs
        predictions = np.argmax(predictions, axis=-1)

        # Step 2: Align and filter predictions/labels
        valid_predictions = []
        valid_labels = []
        for pred, label in zip(predictions, labels):
            mask = label != -100  # Ignore padding tokens
            valid_predictions.append(pred[mask])
            valid_labels.append(label[mask])

        # Step 3: Decode predictions and labels
        decoded_predictions = [
            safe_decode(pred, tokenizer) for pred in valid_predictions
        ]
        decoded_labels = [
            safe_decode(label, tokenizer) for label in valid_labels
        ]

        # Step 4: Log processed predictions and labels for debugging
        print("Processed Predictions (Sample):", decoded_predictions[:5])
        print("Processed Labels (Sample):", decoded_labels[:5])

        # Step 5: Evaluate
        precision, recall, f1, _ = precision_recall_fscore_support(
            decoded_labels, decoded_predictions, average="weighted", zero_division=0
        )
        print("Precision:", precision, "Recall:", recall, "F1:", f1)

        # Save metrics
        metrics = {
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
        }
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        predictions = prediction_output.predictions
        labels = prediction_output.label_ids

        # Step 1: Convert logits to token IDs
        predictions = np.argmax(predictions, axis=-1)

        # Step 2: Align and filter predictions/labels
        valid_predictions = []
        valid_labels = []
        for pred, label in zip(predictions, labels):
            mask = label != -100
            valid_predictions.append(pred[mask])
            valid_labels.append(label[mask])

        # Step 3: Decode predictions and labels
        decoded_predictions = [
            safe_decode(pred, tokenizer) for pred in valid_predictions
        ]
        ground_truths = [
            normalize_output(post_process_prediction(example['output']))
            for example in data_module['predict_dataset']
        ]

        # Step 4: Log processed predictions and labels for debugging
        print("Processed Predictions (Sample):", decoded_predictions[:5])
        print("Processed Ground Truths (Sample):", ground_truths[:5])

        # Step 5: Evaluate
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths, decoded_predictions, average='weighted', zero_division=0
        )
        print("Precision:", precision, "Recall:", recall, "F1:", f1)

        # Save metrics
        prediction_metrics = {
            "predict_precision": precision,
            "predict_recall": recall,
            "predict_f1": f1,
        }
        
        # Save predictions to file

        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = decoded_predictions[i].strip()
                fout.write(json.dumps(example) + '\n')

        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)

    # Save all metrics at the end
    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

# Initialize Weights and Biases for tracking experiments
import wandb
wandb.init(project="llama2_7b_500")
# Placeholder for the main execution logic
if __name__ == "__main__":
    train()