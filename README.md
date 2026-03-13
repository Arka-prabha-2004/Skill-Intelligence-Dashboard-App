# Skill Intelligence Dashboard
*A modular analytics tool for extracting and visualizing skill signals from activity logs.*

## Project Overview

The Skill Intelligence Dashboard is a Streamlit-based analytics dashboard that extracts skill mentions from activity logs and visualizes them using interactive charts and graphs.

The system processes datasets containing tasks, notes, and project descriptions. It detects skill keywords, groups them into hierarchical categories, and displays insights through visual dashboards. The primary goal is to provide a clear understanding of skill distribution and development patterns for team analytics or productivity tracking.

## Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, Regular Expressions (Regex), optional spaCy
- **Visualization:** Plotly and NetworkX
- **Language:** Python

## Features

- **Dynamic Data Ingestion**: Upload custom CSV and Excel datasets directly via the UI, or fallback to default local files.
- **Smart Text Extraction**: Vectorized processing of text logs to extract exact skill mentions based on predefined taxonomy hierarchies.
- **Hierarchical Grouping**: Automatically merges granular skills failing to meet minimum frequency thresholds into parent skill aggregates.
- **Interactive Analytics Engine**:
  - Live Key Performance Indicators (KPIs).
  - Frequency Distribution Bar Charts with CSV data export.
  - Skill Hierarchy Treemaps.
  - Directed NetworkX Relationship Graphs.
  - Grouped Bar Comparisons (General vs Specific skills).
- **Configurable Thresholds**: Sidebar controls to adjust visibility grouping granularity in real-time.

## System Architecture

The Skill Intelligence Dashboard follows a layered, modular architecture. Data flows through dedicated extraction and aggregation layers before being rendered visually in the frontend.

```text
[Dataset Upload]
       │
       ▼
[Dataset Ingestion] (backend/dataset_ingestion.py)
       │
       ▼
[Skill Extraction] (data_processor.py)
       │
       ▼
[Skill Aggregation] (skill_grouper.py)
       │
       ▼
[Streamlit Dashboard] (app.py)
       │
       ▼
[Plotly / NetworkX Visualizations]
```

### Architecture Modules

**Frontend Layer**
- `app.py` (Streamlit dashboard)
  Serves the user interface and coordinates module executions.

**Data Ingestion Layer**
- `backend/dataset_ingestion.py`
  Handles dataset uploads, column normalization, and raw dataset storage.

**Processing Layer**
- `data_processor.py`
  Extracts skills from text using regex-based matching and optional spaCy support.

**Aggregation Layer**
- `skill_grouper.py`
  Groups extracted skills into hierarchical categories and merges low-frequency skills.

**Visualization Layer**
- `Plotly charts`
- `NetworkX graph visualizations`

## Example Workflow

The standard operational pipeline follows this step-by-step path:

1. **Dataset Upload:** The user supplies an activity log CSV/Excel file via the frontend.
2. **Skill Extraction:** The dataset maps through phrase tokens to extract explicit mentions.
3. **Skill Aggregation:** Extracted raw counts are bundled conditionally into parent/child hierarchy mappings based on frequency limitations.
4. **Visualization Dashboard:** The restructured metadata orchestrates visually across multiple KPI metrics and NetworkX/Plotly interface components.

## Dashboard Preview

![Dashboard Preview](assets/dashboard_main.png)

## Installation Instructions

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd Skill-Intelligence-Dashboard-App
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

To start the local dashboard server, run:
```bash
streamlit run app.py
```

The server will initialize and launch locally (typically at `http://localhost:8501`). 

From the dashboard, you can:
1. Upload a new CSV or XLSX dataset via the left sidebar, or let the app default to the existing `activity_log.csv`.
2. Move the **Minimum Skill Frequency** slider to adjust thresholds and watch the graph abstractions simplify in real-time.
3. Export frequency plots using the provided download buttons.

## Project Structure

```text
project_root/
├── app.py                      # Main Streamlit frontend
├── data_processor.py           # NLP and Regex extraction layer
├── skill_grouper.py            # Hierarchical mapping logic
├── requirements.txt            # Dependency definitions
├── status_updates_...csv       # Sample user activity dataset
├── backend/
│   └── dataset_ingestion.py    # IO helpers and normalization
├── docs/                       # Technical architecture documentation
└── assets/                     # Image references and screenshots
```

## Future Improvements

- Implementation of dynamic configuration files (`skill_hierarchy.json` and `dataset_schema.json`) to remove hardcoded application mappings.
- Expanded backend processing pipelines with Parquet serialization to improve memory efficiency on large datasets.
- Schema inference heuristics to adapt dynamically to diverse column names without developer intervention.
- Migration to a lightweight backend API (e.g., FastAPI) to support a dedicated React frontend UI in a more traditional web application production deployment.
