# Gemini Project Context
Skill Intelligence Dashboard

This file provides context and operational rules for Gemini CLI when interacting with this repository.  
Gemini should treat this document as the authoritative description of the project structure, development goals, and coding expectations.

---

# Project Overview

This repository contains a **Skill Intelligence Dashboard** designed to analyze activity logs and extract structured insights about skills demonstrated in work logs.

The system processes datasets (CSV/Excel) that contain notes, tasks, project titles, or descriptions of work performed.  
From this data, the system extracts skill mentions, groups them into hierarchical skill categories, and presents the results in a dashboard.

The primary goal of the system is to produce a **visual representation of skill development and skill distribution** that can be used in analytics, hiring evaluation, or productivity tracking.

The project currently uses **Streamlit** for the dashboard UI and **Python-based processing pipelines** for data extraction and transformation.

---

# Repository Structure

The repository contains the following core files.

app.py  
Streamlit dashboard application responsible for rendering visualizations and interacting with processed data.

data_processor.py  
Handles loading datasets, extracting relevant text, and identifying skill mentions.

skill_grouper.py  
Maps extracted skills into hierarchical skill categories.

requirements.txt  
Project dependencies.

README.md  
General description and instructions for running the project.

Corr.html  
An HTML artifact that visualizes correlations within the dataset. This file is read-only and should not be modified automatically.

---

# Project Objectives

Gemini should prioritize improvements that help the project achieve the following goals.

1. Improve the user interface and usability of the dashboard.
2. Enable the dashboard to support new datasets dynamically.
3. Make the backend architecture modular and maintainable.
4. Introduce configuration-driven skill taxonomy.
5. Provide scalable data ingestion mechanisms.
6. Ensure the system works with unknown dataset schemas.
7. Maintain clear and readable Python code.

---

# Architectural Direction

The system should evolve toward a modular architecture with the following structure.

project_root
│
├── app.py
├── data_processor.py
├── skill_grouper.py
│
├── config
│   ├── skill_hierarchy.json
│   └── dataset_schema.json
│
├── backend
│   └── dataset_ingestion.py
│
├── docs
│   └── architecture.md
│
├── data
│   ├── raw
│   └── processed
│
└── gemini.md

The architecture separates:

UI layer  
Processing layer  
Configuration layer  
Dataset ingestion layer

---

# Coding Principles

Gemini should follow these principles when generating or modifying code.

1. Functions should be small and clearly named.
2. Avoid global state unless absolutely necessary.
3. Prefer pure functions that return data structures instead of printing output.
4. Use pandas for dataset manipulation.
5. Prefer vectorized operations instead of loops where possible.
6. Maintain compatibility with Python 3.10+.

---

# Data Handling Rules

Datasets may vary significantly in structure.

Typical columns may include:

notes  
tasks_completed  
project_title  
description  
next_tasks  
comments  

However, the system must **not assume that these columns exist**.

Instead, Gemini should implement **schema inference** using the following strategy.

1. Identify columns with text-like content.
2. Prefer columns containing keywords such as:

note  
task  
description  
comment  
summary  
title

3. If multiple candidates exist, combine them.

Gemini should produce a suggested dataset schema configuration.

Example dataset_schema.json:

{
  "text_columns": [
    "notes",
    "tasks_completed",
    "description"
  ]
}

---

# Skill Extraction Rules

Skill extraction should follow these rules.

1. Normalize all text.
2. Convert to lowercase.
3. Remove punctuation.
4. Remove excessive whitespace.
5. Tokenize text.
6. Match tokens against known skills.

Skill extraction should output structured data in the following format.

{
  "dataset_id": "example_dataset",
  "inferred_text_columns": ["notes", "tasks_completed"],
  "skill_counts": {
    "python": 15,
    "sql": 9,
    "data visualization": 4
  }
}

This format ensures downstream components can process the results.

---

# Skill Hierarchy

Skills should not remain as flat lists.

Instead they should be grouped hierarchically.

Example structure:

Programming
  Python
  Java
  C++

Data
  SQL
  Data Visualization
  Machine Learning

Soft Skills
  Communication
  Leadership

This hierarchy should be stored in:

config/skill_hierarchy.json

The dashboard should load this configuration dynamically.

---

# Dataset Storage

Uploaded datasets should be stored for reproducibility.

Directory layout:

data/raw/{dataset_id}/original_file.csv  
data/processed/{dataset_id}/processed.parquet

Raw data must never be overwritten.

Processed datasets may be regenerated.

---

# UI Expectations

The dashboard should remain lightweight and intuitive.

Gemini should improve the Streamlit interface with:

KPI cards using st.metric  
Dataset uploader using st.file_uploader  
Download button using st.download_button  
Processing indicators using st.spinner  

Optional filtering widgets in the sidebar.

The UI should prioritize clarity over complexity.

---

# Performance Considerations

Large datasets may contain thousands of rows.

Gemini should:

1. Avoid repeated dataset loading.
2. Use Streamlit caching where appropriate.
3. Prefer vectorized pandas operations.
4. Avoid unnecessary loops.

---

# Token Usage Guidelines

If Gemini generates LLM-assisted extraction code, it must respect token efficiency.

Recommended strategies:

1. Process datasets in chunks.
2. Send only relevant text fields.
3. Request JSON-only responses.
4. Avoid verbose outputs.

---

# Expected Future Enhancements

Gemini may propose improvements such as:

Adding FastAPI backend services  
Creating dataset ingestion pipelines  
Adding configuration-driven skill mappings  
Enhancing dashboard visualizations  
Supporting multiple datasets simultaneously

However, changes should remain incremental and maintain backward compatibility.

---

# Development Philosophy

The repository prioritizes **clarity, extensibility, and robustness** over premature optimization.

Gemini should prefer:

simple architecture  
clear documentation  
minimal dependencies

---

# Instructions for Gemini CLI

When modifying this repository:

1. Explain reasoning briefly before major changes.
2. Suggest file-level edits rather than rewriting the entire project unless explicitly requested.
3. Maintain compatibility with the current codebase.
4. Keep the dashboard runnable at all times.

---

# End of Context