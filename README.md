# Retail Reviews Sentiment Analyser

A Streamlit web app that predicts customer review sentiment using a fine-tuned DeBERTa model.

The app supports CSV upload, batch sentiment prediction, confidence scores, summary statistics, visualisation and CSV export.

## Features

- Upload a CSV file containing customer reviews
- Select the review text column
- Predict sentiment as `negative`, `neutral` or `positive`
- Display prediction confidence scores
- View summary statistics and sentiment distribution charts
- Download prediction results as a CSV file

## Project Structure

```text
project/
├── README.md
├── app.py
├── sentiment_analysis.ipynb
├── Womens Clothing E-Commerce Reviews.csv
├── models/
│   ├── best_model_info.json
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_model/
│       ├── config.json
│       ├── model.safetensors          # download separately
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── thresholds.json
│       └── label_map.json
├── results/
└── app_test_data/
```

## Installation

### 1. Clone or download this repository

```bash
git clone https://github.com/stephenlee66/bc3415-sentiment-analyser.git
cd bc3415-sentiment-analyser
```

### 2. Install required packages

For running the Streamlit app:

```bash
pip install streamlit pandas numpy torch transformers
```

Optional packages for running the Jupyter notebook:

```bash
pip install nltk wordcloud scikit-learn matplotlib seaborn tqdm joblib sentencepiece tiktoken protobuf
```

## Model File Download

The trained DeBERTa weight file is not included in this repository because it is too large for GitHub.

Download `model.safetensors` from the Google Drive link below:

[DeBERTa Model File](https://drive.google.com/file/d/19urlf53FooHujz5r6QyNGzT-g1ewBA7Z/view?usp=sharing)

After downloading, place the file here:

```text
models/bert_model/model.safetensors
```

Without this file, the app will not be able to load the fine-tuned DeBERTa model.

## Run the App

From the project folder, run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser.

## Usage

1. Upload a CSV file containing customer reviews.
2. Select the column that contains the review text.
3. Choose the output format:
   - Add prediction columns to the original CSV
   - Export only the prediction results
4. Click **Analyse Sentiment**.
5. Review the sentiment distribution and confidence scores.
6. Download the results as a CSV file.

## Input CSV Format

Your CSV should contain at least one text column.

Example:

```csv
Review Text,Rating,Product
"Love this dress! Perfect fit.",5,Summer Dress
"Poor quality material.",2,Cotton Shirt
"It's okay, nothing special.",3,Jeans
```

## Output Format

The app adds the following columns:

| Column | Description |
|---|---|
| `Predicted_Sentiment` | Predicted sentiment: negative, neutral or positive |
| `Confidence (%)` | Model confidence score from 0 to 100 |

Example:

```csv
Review Text,Rating,Predicted_Sentiment,Confidence (%)
"Love this dress!",5,positive,96.8
"Poor quality.",2,negative,89.3
"It's okay.",3,neutral,78.5
```

## Troubleshooting

### Model file not found

Make sure `model.safetensors` has been downloaded and placed in with correct file name:

```text
models/bert_model/model.safetensors
```

### App does not start

Check that you are inside the project folder before running:

```bash
streamlit run app.py
```

You can confirm your current folder contains `app.py` by running:

```bash
ls
```