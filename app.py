import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="E-Commerce Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained BERT model and tokenizer"""
    try:
        model_dir = "models/bert_model"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        # Load label mapping
        with open(f"{model_dir}/label_map.json", 'r') as f:
            label_map = json.load(f)
        
        # Create reverse mapping (idx -> label)
        idx_to_label = {v: k for k, v in label_map.items()}

        # Load thresholds
        try:
            with open(f"{model_dir}/thresholds.json", 'r') as f:
                thresholds = json.load(f)
            thresholds = {int(k): v for k, v in thresholds.items()}
        except FileNotFoundError:
            thresholds = None
            st.warning("⚠️ No thresholds file found - using simple argmax prediction instead")    
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, tokenizer, idx_to_label, thresholds, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model is saved in 'models/bert_model/' directory")
        return None, None, None, None

def predict_sentiment(texts, model, tokenizer, idx_to_label, thresholds, device, max_length=256):
    """
    Predict sentiment for a list of texts
    
    Returns:
        predictions: List of predicted sentiment labels
        confidences: List of confidence scores
    """
    predictions = []
    confidences = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(texts):
        # Update progress
        progress = (idx + 1) / len(texts)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(texts)} reviews...")
        
        # Tokenize
        encoding = tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Apply thresholds
            if thresholds is not None:
                # Adjust scores based on thresholds
                adjusted_scores = [probs[i] / thresholds[i] for i in range(len(probs))]
                prediction = np.argmax(adjusted_scores)
            else:
                # Simple argmax (fallback)
                prediction = np.argmax(probs)
            
            # Convert to label
            pred_label = idx_to_label[int(prediction)]
            conf_score = float(probs[prediction])
            
            predictions.append(pred_label)
            confidences.append(round(conf_score * 100, 2))  # Convert to percentage
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return predictions, confidences

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Title and description
    st.title("🛍️ Retail Review Sentiment Analyzer")
    st.markdown("""
    This app uses a fine-tuned **NLP** model to predict sentiment from customer reviews.
    Upload a CSV file with reviews, and get sentiment predictions with confidence scores!
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        **BC3415 Individual Assignment**
        
        **Sentiment Classes:**
        - 🔴 Negative
        - 🟡 Neutral
        - 🟢 Positive
        
        **Features:**
        - Batch Prediction
        - Confidence Scores
        - CSV Export
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("""
        1. Upload CSV file
        2. Select text column
        3. Click 'Analyze Sentiment'
        4. Download results
        """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, idx_to_label, thresholds, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File upload
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer reviews"
    )
    
    if uploaded_file is not None:
        # Read CSV
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', 
                       unsafe_allow_html=True)
            
            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), width='stretch')
            st.caption(f"Total rows: {len(df)}")
            
            # Column selection
            st.subheader("⚙️ Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Select the column containing review text:",
                    options=df.columns.tolist(),
                    help="Choose the column with customer reviews"
                )
            
            with col2:
                output_format = st.radio(
                    "Output format:",
                    ["Add columns to original CSV", "New CSV with only results"],
                    help="Choose how to format the output"
                )
            
            # Analyze button
            if st.button("🚀 Analyze Sentiment", type="primary"):
                
                # Validate text column
                if text_column not in df.columns:
                    st.error(f"Column '{text_column}' not found in the uploaded file.")
                    return
                
                # Check for missing values
                missing_count = df[text_column].isna().sum()
                if missing_count > 0:
                    st.warning(f"⚠️ Found {missing_count} missing values. They will be skipped.")
                    df = df.dropna(subset=[text_column])
                
                # Get texts
                texts = df[text_column].tolist()
                
                st.info(f"🔍 Analyzing {len(texts)} reviews...")
                
                # Predict
                try:
                    predictions, confidences = predict_sentiment(
                        texts, model, tokenizer, idx_to_label, thresholds, device
                    )
                    
                    # Create results dataframe
                    if output_format == "Add columns to original CSV":
                        result_df = df.copy()
                        result_df['Predicted_Sentiment'] = predictions
                        result_df['Confidence (%)'] = confidences
                    else:
                        result_df = pd.DataFrame({
                            'Review_Text': texts,
                            'Predicted_Sentiment': predictions,
                            'Confidence (%)': confidences
                        })
                    
                    # Display results
                    st.success("✅ Analysis complete!")
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    sentiment_counts = pd.Series(predictions).value_counts()
                    
                    with col1:
                        st.metric("Total Reviews", len(predictions))
                    with col2:
                        st.metric("🔴 Negative", sentiment_counts.get('negative', 0))
                    with col3:
                        st.metric("🟡 Neutral", sentiment_counts.get('neutral', 0))
                    with col4:
                        st.metric("🟢 Positive", sentiment_counts.get('positive', 0))
                    
                    # Average confidence
                    avg_confidence = sum(confidences) / len(confidences)
                    st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                    
                    # Results preview
                    st.subheader("📋 Results Preview")
                    st.dataframe(result_df.head(20), width='stretch')
                    
                    # Sentiment distribution chart
                    st.subheader("📊 Sentiment Distribution")
                    sentiment_df = pd.DataFrame({
                        'Sentiment': [s.capitalize() for s in sentiment_counts.index],
                        'Count': list(sentiment_counts.values)
                    })
                    st.bar_chart(sentiment_df.set_index('Sentiment'))
                    
                    # Download button
                    st.subheader("💾 Download Results")
                    
                    csv = convert_df_to_csv(result_df)
                    
                    st.download_button(
                        label="📥 Download CSV with Predictions",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        help="Click to download the results as CSV"
                    )
                    
                    st.markdown('<div class="success-box">✅ You can now download your results!</div>', 
                               unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")
    
    else:
        # Show example
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **📝 CSV Format Example:**
        
        Your CSV should have at least one column with text reviews:
        
        | Review Text | Rating |
        |-------------|--------|
        | Love this dress! Perfect fit and quality. | 5 |
        | The material feels cheap and it runs small. | 2 |
        | It's okay, nothing special. | 3 |
        
        Upload your CSV file to get started!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
