# **GEE\! Grammar Error Explanation with Large Language Models**

This repository supports the **Reproducibility Study** for the NAACL 2024 paper **"GEE\! Grammar Error Explanation with Large Language Models"** by Yixiao Song et al. The study explores a novel Grammar Error Explanation (GEE) task that combines grammatical corrections with natural language explanations, bridging the gap in existing Grammar Error Correction (GEC) systems.

---

## **Features**

* **LoRA-based Fine-Tuning**: Efficient memory optimization for large language models.  
* **Multilingual Support**: Tested on German, Chinese, and English datasets.  
* **Custom Metrics**: Precision, Recall, and F1-score evaluation.  
* **Reproducibility**: Consistent results with seed management and detailed logging.  
* **Quantization Support**: Enables 4-bit and 8-bit quantization for efficient inference.  
* **Experiment Tracking**: Integrated with Weights & Biases (W\&B) for logging.

---

## **Setup**

### **Prerequisites**

* Python 3.8 or above  
* CUDA-compatible GPU  
* Hugging Face Transformers Library  
* Additional Python libraries 

### **Installation**

1. Clone the repository:  
   `git clone Our GitHub link is https://github.com/Diwita19/GEE-with-LLMs.git`

2. Install dependencies:  
   `pip install torch transformers bitsandbytes datasets evaluate pandas scikit-learn tqdm wandb`

3. Required Libraries:
    Below is a list of libraries required to run the project:
      - ***General Utilities***:
        - `collections`
        - `copy`
        - `json`
        - `os`
        - `argparse`
        - `logging`
        - `warnings`
        - `dataclasses`
      - ***Data Handling***:
        - `numpy`
        - `pandas`
        - `tqdm`
        - `packaging`
      - ***Machine Learning Frameworks***:
        - `torch` (PyTorch)
        - `transformers` (Hugging Face Transformers)
        - `bitsandbytes`
      - ***Dataset Handling***:
        - `datasets` (Hugging Face Datasets)
        - `evaluate`
      - ***Scikit-learn Metrics***:
        - `sklearn.metrics`
      - ***Parameter-Efficient Fine-Tuning (PEFT)***:
        - `peft`
      - ***Experiment Tracking***:
        - `wandb`
        
      To install all required packages, run in bash:
      pip install torch transformers bitsandbytes datasets evaluate pandas scikit-learn tqdm wandb

---

## **File Structure**

1. **'Prompts'**:

  * **'de_gpt4_end2end_prompt_utils.py'**: prompts used for Section 3 in the paper
  * **'de_prompt_utils.py'**: prompts for German atomic edit extraction and explanation generation
  * **'zh_prompt_utils.py'**: prompts for Chinese atomic edit extraction and explanation generation

2. **'fine-tune_llama2-7b'**:

  * **'fine-tune_llama2-7b.sh'**: Parameters for fine-tuning the model
  * **'qlora.py'**: Source code

3. **'rule_based_screening.py'**: Heuristic rules for screening out low-level mistakes in atomic edit extraction
4. **'SequenceMatcher_rough_edits.py'**: Uses SequenceMatcher from difflib to extract rough edits

---

## **Dataset Preparation**

### **fine-tune_data:**

The training and test data of LLM fine-tuning for German and Chinese atomic edit extraction. The data is in the format for fine-tuning ChatGPT. Sentence pair is the source and target sentence; list of edits are the rough edits extracted by SequenceMatcher; list of labels are the labels of the edits; content is the gold atomic edits.

### **human_annotation_data**: 

The anonymized raw human annotation data

### **Supported Formats:**

* **JSONL**: Each line is a JSON object containing user and assistant messages.  
* **CSV/TSV**: Tabular formats with input-output columns.  
* **JSON**: Nested structure with instructions and responses.

### **Preprocessing:**

The script handles padding, truncation, and token alignment for optimal model performance.

---

## **Evaluation Metrics**

Evaluation is conducted using:

* Precision  
* Recall  
* F1-Score

These metrics are calculated using `sklearn.metrics.precision_recall_fscore_support`.
 

The results highlight the robustness of the pipeline, with detailed analyses of errors, including challenges with noisy inputs and semantic errors.