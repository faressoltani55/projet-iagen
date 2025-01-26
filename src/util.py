from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import os

# Define the model names
gpt2_model = "gpt2"
gpt_neo_model = "EleutherAI/gpt-neo-125M"

# Define the local directory paths
gpt2_local_path = f"models/{gpt2_model}"
gpt_neo_local_path = f"models/{gpt_neo_model}"

# Load GPT-2 tokenizer and model from local path or Hugging Face
if os.path.exists(gpt2_local_path):
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_local_path, from_tf=True)
    gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_local_path, from_tf=True)
else:
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model, from_tf=True)
    gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model, from_tf=True)

# Load GPT-Neo tokenizer and model from local path or Hugging Face
if os.path.exists(gpt_neo_local_path):
    gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_local_path, from_tf=True)
    gpt_neo = AutoModelForCausalLM.from_pretrained(gpt_neo_local_path, from_tf=True)
else:
    gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_model)
    gpt_neo = AutoModelForCausalLM.from_pretrained(gpt_neo_model)


def create_poem_prompt(keyword, model_name):
    if model_name == "gpt2":
        return keyword
    else:
        return f"Let me tell you about {keyword}."


def generate_text(model, tokenizer, prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_similarity(text1, text2):
    bleu_score = sentence_bleu([text1.split()], text2.split())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(text1, text2)
    return bleu_score, rouge_scores
