# System Architecture Report
## Skill Intelligence Dashboard

This document describes the architecture, data flow, and design decisions behind the Skill Intelligence Dashboard.  
It outlines the responsibilities of each module, identifies current system limitations, and proposes improvements for a more scalable and modular architecture.

## Table of Contents

1. System Overview  
2. Module Responsibilities  
3. Data Flow  
4. Current Limitations  
5. Refactor Opportunities  
6. Future Architecture  
7. Processing Pipeline  
8. Architecture Principles

## 1. System Overview
The **Skill Intelligence Dashboard** is a Streamlit-based analytics application designed to analyze activity logs and extract structured insights about skills demonstrated in work records.

The system processes datasets containing work logs such as tasks, notes, or project descriptions. It scans these textual entries for references to known technical skills, aggregates their occurrences, groups them into hierarchical skill categories, and visualizes the results through an interactive dashboard.

The primary objective is to provide a **visual understanding of skill distribution and skill development patterns**, which can support analytics, productivity insights, or hiring evaluations.

## 2. Module Responsibilities
- **`app.py`** – Presentation layer.  
  Implements the Streamlit dashboard interface, handles sidebar configuration (such as frequency thresholds), and renders visualizations including KPI metrics, bar charts, treemaps, and network graphs using Plotly and NetworkX.
- **`data_processor.py`** – Data processing layer.  
  Responsible for loading datasets using the Pandas library, extracting relevant text fields, and detecting occurrences of predefined skills defined in the `SKILL_HIERARCHY`.
- **`skill_grouper.py`** – Aggregation layer.  
  Groups detected skills into hierarchical categories and applies frequency thresholds. Skills exceeding the threshold remain visible individually while still contributing to their parent category totals.
- **`README.md`** – Project documentation.  
  Provides an overview of the system, its features, and instructions for running the application locally.
- **`requirements.txt`** – Dependency specification.  
  Lists the Python libraries required to run the dashboard, including packages such as `streamlit`, `pandas`, `plotly`, `networkx`, and `spacy`.

## 3. Data Flow
1. **Dataset Loading**  
   A dataset (currently `activity_log.csv`) is loaded into memory using `pandas.read_csv()` within `data_processor.py`.
2. **Text Field Extraction**  
   Relevant text columns such as `tasks_completed`, `next_tasks`, `notes`, and `project_title` are selected for analysis.
3. **Skill Detection**  
   Text is normalized and scanned using keyword and phrase matching against the predefined set of skills defined in the `SKILL_HIERARCHY`.
4. **Skill Counting**  
   Each detected skill occurrence is recorded, producing a flat dictionary of skill frequencies.
5. **Hierarchical Grouping**  
   The `skill_grouper.py` module aggregates these counts and groups them under parent categories according to the hierarchy.
6. **Visualization**  
   The processed skill data is passed to `app.py`, which renders metrics and charts including bar graphs, treemaps, and network visualizations.

## 4. Current Limitations

- **Hardcoded Skill Taxonomy**  
  The `SKILL_HIERARCHY` is defined directly inside `data_processor.py`. Updating skill definitions currently requires modifying source code.
- **Rigid Dataset Schema**  
  The system assumes the presence of specific column names such as `tasks_completed` or `notes`. Datasets with different schemas require manual adjustments.
- **Limited Scalability**  
  The application processes entire datasets in memory and performs row-wise iteration, which can become inefficient for large datasets.
- **No Processed Data Caching**  
  All skill extraction operations are recomputed on every application run instead of using cached or serialized intermediate results.

## 5. Refactor Opportunities
- **Configuration-Driven Skill Taxonomy**  
  Move the `SKILL_HIERARCHY` into an external configuration file (`config/skill_hierarchy.json`) so skill definitions can be updated without modifying source code.
- **Dynamic Dataset Uploading**  
  Add a `st.file_uploader` component to allow users to upload datasets directly through the dashboard.
- **Schema Inference Mechanism**  
  Implement heuristics to automatically detect text columns using keywords such as `task`, `description`, or `notes`.
- **Vectorized Text Processing**  
  Replace row-wise loops with Pandas vectorized string operations (`str.contains`, `str.extract`) to improve performance.
- **Processed Data Caching**  
  Store processed results in a serialized format such as Parquet in a `data/processed/` directory to avoid recomputation.

## 6. Future Architecture
To improve scalability, maintainability, and modularity, the dashboard should evolve toward the following architecture:

```text
project_root
│
├── app.py                     # Presentation Layer (Streamlit)
├── data_processor.py          # Backend Processing Pipeline
├── skill_grouper.py           # Backend Processing Pipeline
│
├── config/                    # Configuration Layer
│   ├── skill_hierarchy.json   # Externalized skill taxonomy
│   └── dataset_schema.json    # Heuristics for dynamic schema inference
│
├── backend/
│   └── dataset_ingestion.py   # Dataset Ingestion Module
│
├── docs/
│   └── architecture.md        # Technical Documentation
│
├── data/
│   ├── raw/                   # Immutable raw dataset uploads
│   └── processed/             # Regenerable processed (.parquet) extracts
│
└── requirements.txt
```

### Future Architecture Components

The system can be improved by separating responsibilities into clear layers: dataset ingestion, configuration management, processing logic, and visualization.

- **Dataset Ingestion Module**: Responsible for safely managing file uploads, dynamically sniffing out text headers, and persisting raw datasets.
- **Configuration-Driven Skill Taxonomy**: All skill structures and regex rules are parsed dynamically from JSON at runtime, ensuring strict separation of concerns.
- **Backend Processing Pipeline**: Designed to ingest raw datasets in chunks, map variables vectorially, and emit optimized `.parquet` outputs to `data/processed/`.
- **UI Layer (`app.py`)**: Responsible only for orchestrating inputs and mapping pre-processed datasets to visuals (e.g., `st.metric`, Plotly).

## 7. Processing Pipeline

```
Raw Dataset
     ↓
Dataset Ingestion Module
     ↓
Schema Inference
     ↓
Skill Extraction
     ↓
Skill Grouping
     ↓
Processed Skill Dataset
     ↓
Dashboard Visualization
```

## 8. Architecture Principles

The architecture of the system follows several guiding principles:

- **Separation of Concerns** – UI, data ingestion, and processing logic are separated into independent modules.
- **Configurability** – Skill hierarchies and dataset schemas are defined in configuration files instead of hardcoded values.
- **Scalability** – The architecture supports larger datasets through caching and vectorized data processing.
- **Extensibility** – New datasets, skill definitions, or visualizations can be added without restructuring the core system.
- **Reproducibility** – Raw datasets are preserved separately from processed outputs to ensure results can be reproduced.

This architecture provides a foundation for evolving the Skill Intelligence Dashboard from a single-dataset analytics tool into a scalable and extensible skill intelligence platform.