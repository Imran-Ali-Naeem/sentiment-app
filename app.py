# sentiment_analysis_app.py
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        background-color: #17a2b8;
        height: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #6c757d;
    }
    .flag-sarcasm {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .flag-lowconf {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer from Hugging Face Hub"""
    try:
        # Your Hugging Face model repository
        MODEL_REPO = "ImranAliNaeem/bert-sentiment-analysis"
        
        st.sidebar.info("🔄 Loading model from Hugging Face Hub...")
        
        # Load model and tokenizer from Hugging Face
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        
        # Create label mappings based on model configuration
        num_labels = model.config.num_labels
        
        # Check if model has label names in config
        if hasattr(model.config, 'id2label') and model.config.id2label:
            label_map = {}
            for key, value in model.config.id2label.items():
                label_map[int(key)] = value
            label_map_reverse = {v: k for k, v in label_map.items()}
        else:
            # Default to binary sentiment labels
            if num_labels == 2:
                label_map = {0: "Negative", 1: "Positive"}
                label_map_reverse = {"Negative": 0, "Positive": 1}
            else:
                # Generic labels for multi-class
                label_map = {i: f"Class_{i}" for i in range(num_labels)}
                label_map_reverse = {f"Class_{i}": i for i in range(num_labels)}
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model, tokenizer, label_map, label_map_reverse
        
    except Exception as e:
        st.error(f"❌ Error loading model from Hugging Face: {str(e)}")
        st.info("🔧 Troubleshooting tips:")
        st.info("1. Check your internet connection")
        st.info("2. Verify the model repository exists: https://huggingface.co/ImranAliNaeem/bert-sentiment-analysis")
        st.info("3. Make sure all model files are uploaded to Hugging Face")
        return None, None, None, None

def is_sarcastic(text):
    """
    Detect sarcasm based on positive words in negative context.
    """
    text_lower = text.lower()
    positive_words = ['great', 'fantastic', 'wonderful', 'brilliant', 'perfect', 'love', 'excellent', 'awesome']
    negative_context = ['broken', 'failed', 'problem', 'waiting', 'terrible', 'horrible', 'bad', 'waste', 'disappointed', 'useless']
    
    has_positive = any(word in text_lower for word in positive_words)
    has_negative = any(word in text_lower for word in negative_context)
    
    return has_positive and has_negative

def predict_sentiment(text: str, model, tokenizer, label_map, max_length=128):
    """
    Enhanced sentiment prediction with sarcasm detection and confidence thresholds.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize input text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get sentiment label
    predicted_sentiment = label_map[predicted_class]
    
    # Initialize flags
    flags = []
    
    # Enhanced logic: Handle sarcasm and low confidence
    # 1. Sarcasm detection
    if is_sarcastic(text):
        predicted_sentiment = "Negative"  # Override for sarcasm
        confidence = max(0.7, confidence)  # Ensure reasonable confidence
        flags.append('sarcasm_detected')
    
    # 2. Low confidence threshold
    elif confidence < 0.75:
        flags.append('low_confidence')
    
    # Get all probabilities
    all_probabilities = {}
    for i in range(len(probabilities[0])):
        label_name = label_map[i]
        all_probabilities[label_name] = probabilities[0][i].item()
    
    return {
        'text': text,
        'sentiment': predicted_sentiment,
        'confidence': confidence,
        'probabilities': all_probabilities,
        'predicted_class': predicted_class,
        'flags': flags
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">😊 😞 Sentiment Analysis App</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a fine-tuned BERT model with enhanced sarcasm detection to analyze sentiment in text."
    )
    
    # Load model
    with st.spinner("Loading enhanced sentiment analysis model from Hugging Face Hub..."):
        model, tokenizer, label_map, label_map_reverse = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load the model. Please check the troubleshooting tips above.")
        return
    
    # Model Info
    st.sidebar.title("Model Info")
    st.sidebar.metric("Model", "BERT Fine-tuned")
    st.sidebar.metric("Labels", f"{len(label_map)} classes")
    st.sidebar.metric("Features", "Sarcasm Detection ✅")
    
    # Debug info (collapsible)
    with st.sidebar.expander("Debug Info"):
        st.write("Label Map:", label_map)
        st.write("Model Repository:", "ImranAliNaeem/bert-sentiment-analysis")
    
    # Enhanced examples including sarcasm
    st.sidebar.title("Try These Examples")
    enhanced_examples = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced.",
        "Fantastic, now everything is broken.",  # Sarcasm
        "Excellent customer service and fast delivery!",
        "Wonderful, I love waiting for hours.",  # Sarcasm
        "It's okay, nothing special but gets the job done."
    ]
    
    for example in enhanced_examples:
        if st.sidebar.button(example[:50] + "..." if len(example) > 50 else example):
            st.session_state.input_text = example
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text for Analysis")
        
        # Text input
        input_text = st.text_area(
            "Type your text here:",
            height=150,
            placeholder="Enter your text here to analyze its sentiment...",
            value=st.session_state.get('input_text', ''),
            key="main_input"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Analyzing sentiment with enhanced detection..."):
                    result = predict_sentiment(input_text, model, tokenizer, label_map)
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Determine sentiment display
                sentiment_text = result['sentiment'].lower()
                if 'positive' in sentiment_text:
                    sentiment_emoji = "😊"
                    sentiment_display = "Positive"
                    sentiment_class = "positive"
                else:
                    sentiment_emoji = "😞"
                    sentiment_display = "Negative" 
                    sentiment_class = "negative"
                
                # Confidence
                confidence_percent = result['confidence'] * 100
                
                # Display sentiment with flags
                sentiment_html = f'<div class="{sentiment_class}"><h3>{sentiment_emoji} {sentiment_display} Sentiment'
                
                # Add flags if present
                if result['flags']:
                    flags_html = ""
                    for flag in result['flags']:
                        if flag == 'sarcasm_detected':
                            flags_html += ' <span class="flag-sarcasm">🚩 Sarcasm Detected</span>'
                        elif flag == 'low_confidence':
                            flags_html += ' <span class="flag-lowconf">⚠️ Low Confidence</span>'
                    sentiment_html += flags_html
                
                sentiment_html += '</h3></div>'
                st.markdown(sentiment_html, unsafe_allow_html=True)
                
                # Confidence with progress bar
                st.subheader(f"🎯 Confidence: {confidence_percent:.1f}%")
                st.progress(result['confidence'])
                
                # Probabilities
                st.subheader("📈 Detailed Probabilities")
                
                # Create columns for probability bars
                prob_cols = st.columns(len(result['probabilities']))
                
                for idx, (label, prob) in enumerate(result['probabilities'].items()):
                    with prob_cols[idx]:
                        # Determine display label
                        label_lower = label.lower()
                        if 'positive' in label_lower:
                            display_label = "Positive"
                            emoji = "😊"
                        else:
                            display_label = "Negative"
                            emoji = "😞"
                            
                        st.metric(
                            label=f"{emoji} {display_label}",
                            value=f"{prob:.1%}",
                            delta=None
                        )
                
                # Flag explanations
                if result['flags']:
                    st.subheader("🔍 Analysis Notes")
                    for flag in result['flags']:
                        if flag == 'sarcasm_detected':
                            st.info("🤔 **Sarcasm Detected**: The text contains positive words in a negative context, which has been interpreted as sarcasm.")
                        elif flag == 'low_confidence':
                            st.warning("⚠️ **Low Confidence**: The sentiment is unclear. The text may be neutral or contain mixed emotions.")
                
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.subheader("⚡ Quick Analysis")
        
        # Batch analysis
        st.write("Analyze multiple texts (one per line):")
        batch_text = st.text_area(
            "Batch texts:",
            height=200,
            placeholder="Enter multiple texts, one per line...",
            key="batch_text"
        )
        
        if st.button("🔍 Analyze Batch", use_container_width=True):
            if batch_text.strip():
                texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, text in enumerate(texts):
                    status_text.text(f"Analyzing {i+1}/{len(texts)}: {text[:50]}...")
                    result = predict_sentiment(text, model, tokenizer, label_map)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(texts))
                
                status_text.text("✅ Analysis complete!")
                
                # Display batch results
                st.subheader("📦 Batch Results")
                
                # Count sentiments and flags
                positive_count = sum(1 for r in results if 'positive' in r['sentiment'].lower())
                negative_count = len(results) - positive_count
                sarcasm_count = sum(1 for r in results if 'sarcasm_detected' in r['flags'])
                low_conf_count = sum(1 for r in results if 'low_confidence' in r['flags'])
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("😊 Positive", positive_count)
                with col2:
                    st.metric("😞 Negative", negative_count)
                with col3:
                    st.metric("🎭 Sarcasm", sarcasm_count)
                with col4:
                    st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                for i, result in enumerate(results, 1):
                    # Determine sentiment display
                    sentiment_text = result['sentiment'].lower()
                    if 'positive' in sentiment_text:
                        sentiment_emoji = "😊"
                        sentiment_display = "Positive"
                    else:
                        sentiment_emoji = "😞" 
                        sentiment_display = "Negative"
                    
                    # Add flag indicators
                    flag_emojis = ""
                    if 'sarcasm_detected' in result['flags']:
                        flag_emojis += " 🎭"
                    if 'low_confidence' in result['flags']:
                        flag_emojis += " ⚠️"
                    
                    with st.expander(f"{i}. {sentiment_emoji} {sentiment_display} ({result['confidence']:.1%}){flag_emojis}"):
                        st.write(f"**Text:** {result['text']}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        
                        if result['flags']:
                            st.write(f"**Flags:** {', '.join(result['flags'])}")
                        
                        # Probability breakdown
                        prob_text = " | ".join([
                            f"{'Positive' if 'positive' in k.lower() else 'Negative'}: {v:.2%}" 
                            for k, v in result['probabilities'].items()
                        ])
                        st.write(f"**Probabilities:** {prob_text}")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Enhanced Model Details:** Fine-tuned BERT with sarcasm detection | "
        "**Hosted on:** Hugging Face Hub | "
        "**Repository:** [ImranAliNaeem/bert-sentiment-analysis](https://huggingface.co/ImranAliNaeem/bert-sentiment-analysis)"
    )

if __name__ == "__main__":
    main()
