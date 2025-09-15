from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os


def extract_and_rephrase(text, instruction):
    prompt = f"""
    Инструкция: {instruction}
    Текст: {text}
    Результат:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.3,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Загрузка модели
if not os.path.isdir('./models/llm/models--google--flan-t5-base'):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models/llm')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models/llm')