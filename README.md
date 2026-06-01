# Retail Reviews Sentiment Analyzer

A Streamlit web application demo for analyzing customer review sentiment using a fine-tuned DeBERTa model.

## Features

✅ **CSV Upload** - Upload files with customer reviews
✅ **Batch Prediction** - Analyze multiple reviews at once
✅ **Confidence Scores** - Get prediction confidence percentages
✅ **Interactive UI** - User-friendly interface with progress tracking
✅ **CSV Export** - Download results with predictions
✅ **Summary Statistics** - View sentiment distribution and metrics
✅ **Visualization** - Bar charts for sentiment analysis

## Installation

### Prerequisites

- **CPU only users:** Python 3.8 or above
- **GPU users (NVIDIA):** Python 3.10 or 3.11 recommended

### Step 1: Install Dependencies

Install the required packages to run the Streamlit app:

```bash
pip install streamlit pandas numpy torch transformers
```

Optional if you also want to run the Jupyter notebook (sentiment_analysis.ipynb):

Non-GPU users might have to switch to lighter versions of bert e.g. distilbert

```bash
pip install pandas numpy nltk wordcloud scikit-learn torch transformers matplotlib seaborn tqdm joblib sentencepiece tiktoken protobuf
```

For GPU Users:

Use Python 3.10 or 3.11 in a virtual environment and install the CUDA-enabled version of PyTorch for GPU acceleration.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Verify Project Directory & Model File Download

Due to Blackboard upload size limitations, the trained model file model.safetensors is hosted separately at:

https://drive.google.com/file/d/19urlf53FooHujz5r6QyNGzT-g1ewBA7Z/view?usp=sharing

Please download model.safetensors and place it in the folder below before running the application.

```
project/
├── README.md                               # usage guide
├── app.py                                  # streamlit app (run this only)
├── sentiment_analysis.ipynb                # full ml pipeline
├── Womens Clothing E-Commerce Reviews.csv  # training data
├── models/                                 # model storage
│   └── bert_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── thresholds.json
│       └── label_map.json
├── results/                                # visuals from ipynb
├── app_test_data/                          # sample test data
├── Stephen_Demo.mp4                        # demo vid
```

### Step 3: Change to Project Directory

Open Terminal or VS Code terminal and navigate to your project folder:

```bash
cd /path/to/project                         # your own file path
```
Replace /path/to/project with your actual folder path.

To confirm you are in the right folder:

```bash
ls
```

You should see files like README.md, app.py, etc.

### Step 4: Run the App

```bash
streamlit run app.py
```

## Usage Guide

### Step 1: Upload CSV File

1. Click "Browse files" button
2. Select your CSV file containing reviews
3. The file will be previewed automatically

### Step 2: Select Text Column

1. Choose the column containing review text from dropdown
2. Select output format preference:
   - **Add columns to original CSV**: Adds sentiment columns to your data
   - **New CSV with only results**: Creates a clean output with just results

### Step 3: Analyze Sentiment

1. Click "🚀 Analyze Sentiment" button
2. Wait for processing (progress bar will show)
3. View results and statistics

### Step 4: Download Results

1. Review the sentiment distribution and statistics
2. Click "📥 Download CSV with Predictions"
3. Save the output file

## CSV Format

Your input CSV should have at least one column with text:

```csv
Review Text,Rating,Product
"Love this dress! Perfect fit.",5,Summer Dress
"Poor quality material.",2,Cotton Shirt
"It's okay, nothing special.",3,Jeans
```

## Output Format

The app adds two columns to your data:

- **Predicted_Sentiment**: negative, neutral or positive
- **Confidence (%)**: Model confidence percentage (0-100)

Example output:

```csv
Review Text,Rating,Predicted_Sentiment,Confidence (%)
"Love this dress!",5,positive,96.8
"Poor quality.",2,negative,89.3
"It's okay.",3,neutral,78.5
```

## Troubleshooting

### Model Not Found Error

**Solution:** Make sure the trained model files are present in `models/bert_model/`.

## Credits

Built by Stephen with ❤️ using:
- Streamlit
- HuggingFace Transformers
- PyTorch