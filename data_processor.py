import pandas as pd
import spacy
import re
from collections import Counter
from backend.dataset_ingestion import load_dataset

# Try loading spacy, fallback to simple regex if missing
# Note: spaCy is optional and not strictly required for the core regex-based keyword extraction.
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

SKILL_HIERARCHY = {
    "Large Language Models": ["qwen", "gemma", "gpt", "chatgpt", "bert", "transformers", "llm"],
    "Model Optimization": ["quantization", "int8", "fp16", "lora", "dpo", "finetuning"],
    "Prompt Engineering": ["prompting", "few-shot", "zero-shot", "chain-of-thought", "cot"],
    "Evaluation Metrics": ["llm judge", "f1 score", "ner", "ontonotes5", "ontonotes"],
    "Infrastructure Tools": ["colab", "gpu", "t4 gpu", "huggingface", "python", "streamlit", "react"],
    "Science and Math": ["kinematics", "calculus", "arithmetic", "physics"]
}

# Flatten for extraction
ALL_SKILLS = set()
for parent, children in SKILL_HIERARCHY.items():
    ALL_SKILLS.add(parent.lower())
    for child in children:
        ALL_SKILLS.add(child.lower())

# Sort skills by length descending to match longest phrases first (e.g. "llm judge" before "llm")
# This is computed once globally instead of inside the extraction loop for performance.
SORTED_SKILLS = sorted(list(ALL_SKILLS), key=len, reverse=True)

def extract_skills_from_text(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    found_skills = []
    
    # Simple regex based matching for exact phrases
    
    # We replace found skills with spaces so we don't double count "llm" inside "llm judge"
    for skill in SORTED_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = re.findall(pattern, text)
        if matches:
            found_skills.extend([skill] * len(matches))
            text = re.sub(pattern, ' ', text)
            
    return found_skills

def extract_skills(df):
    text_cols = ['tasks_completed', 'next_tasks', 'notes', 'project_title']
    all_extracted = []
    
    for col in text_cols:
        if col in df.columns:
            # Use dropna() to avoid iterating over missing values
            for text in df[col].dropna():
                all_extracted.extend(extract_skills_from_text(text))
                
    skill_counts = Counter(all_extracted)
    return dict(skill_counts)

def process_dataset(data):
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = load_dataset(data)
    
    return extract_skills(df)

def process_logs(file_path="activity_log.csv"):
    return process_dataset(file_path)

if __name__ == "__main__":
    counts = process_logs("activity_log.csv")
    print("Extracted Skills:", counts)
