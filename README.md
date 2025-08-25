# Census Income Prediction Project

This project implements machine learning models to predict whether a person's income exceeds $50K based on census data. The project includes data preprocessing, multiple classifier training (Logistic Regression, Random Forest, XGBoost), and comprehensive analysis tools.

### [Project Report](https://github.com/Ceiceiceii/census_income_classification/blob/main/ProjectReport.pdf)

## Table of Contents

- [Setup Environment](#setup-environment)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Model Demo](#model-demo)
- [Analysis Tools](#analysis-tools)
- [Dependencies](#dependencies)

## Setup Environment

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Required Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install core ML packages
pip install pandas numpy scikit-learn xgboost

# Install visualization packages
pip install matplotlib seaborn

# Install additional utilities
pip install scipy jupyter

# Optional: Install all at once
pip install -r requirement.txt
```


## Project Structure

```
census_classification_project/
├── README.md                           # This file
├── ProjectReport.pdf                   # Project Report
├── convert_to_csv.py                   # Data conversion script
├── simple_downsampled_classifier.py    # Main training script
├── model_demo.py                       # Model demonstration script
├── data_exploration.py                 # EDA script in python
├── occupation_3d_clustering.py         # Occupation analysis
├── census_data.csv                     # Raw data (generated)
├── simple_downsampled_model.pkl        # Trained model (generated)
├── segmentation_plot                   # 3D clustering plot (generated)
└── eda_plots/                          # EDA visualizations (generated)

```

## Training/Data Exploration/Segementation Pipeline

### Step 1: Data Preparation

First, convert the raw data to CSV format (if the .data and .columns files are not in /source_data, please paste the data file in first):

```bash
python convert_to_csv.py
```

**What this does:**
- Converts raw census data to CSV format
- Creates `census_data.csv` file
- Handles data formatting and basic preprocessing


### Step 2: Train Models

Run the main training script:

```bash
python simple_downsampled_classifier.py
```

**What this does:**
- Loads and preprocesses census data
- Handles outliers using mode-based replacement
- Performs class balancing through downsampling
- Trains three models: Logistic Regression, Random Forest, XGBoost
- Evaluates models using accuracy, ROC-AUC, and PR-AUC
- Saves the best performing model to `simple_downsampled_model.pkl`


## Model Demo

### If Model Already Exists

If you have a pre-trained model (`simple_downsampled_model.pkl`), you can run the demo directly:

```bash
python model_demo.py
```

### If No Model Exists

The demo script will automatically guide you:

```bash
python model_demo.py
# Output: Please run simple_downsampled_classifier.py first to train a model.
```

**What the demo does:**
- Loads the trained model
- Demonstrates predictions on sample data
- Shows model performance metrics
- Provides interactive examples


## Analysis Tools

### Exploratory Data Analysis

```bash
python data_exploration.py
```

**Features:**
- Comprehensive statistical analysis
- Missing value detection
- Feature correlation analysis
- Target variable distribution
- Visualization plots saved to `eda_plots/` folder

### Occupation-Based Clustering Analysis

```bash
python occupation_3d_clustering.py
```

**Features:**
- 3D visualization of 15 major occupation codes
- Customer segmentation analysis
- Retail marketing insights
- Group difference analysis

### Performance Tips

1. **For faster training:** Reduce `n_estimators` in RandomForest and XGBoost
2. **For better accuracy:** Increase `n_estimators` but expect longer training time
3. **For memory efficiency:** Use `n_jobs=1` instead of `n_jobs=-1`

## Model Performance

Performance varies based on:
- Data preprocessing choices
- Feature selection
- Hyperparameter tuning
- Class balance handling

---

**Quick Start Command Sequence:**
```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt

# Train
python convert_to_csv.py
python simple_downsampled_classifier.py

# Demo
python model_demo.py

# Analysis (optional)
python data_exploration.py
python occupation_3d_clustering.py
```
