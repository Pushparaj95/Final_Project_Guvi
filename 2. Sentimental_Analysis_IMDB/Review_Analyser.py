import streamlit as st
import streamlit.components.v1 as components
from IPython.display import HTML, display
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import json
import torch
import torch.nn as nn
import re

vocab_size = 5000
output_size = 1
embedding_size = 256
hidden_size = 512
n_layers = 2
dropout=0.3
english_stopwords = set(stopwords.words('english'))


def tokenize(text):
    """
    Method to tokenize the text into words
    """
    return word_tokenize(text)

def clean_pipeline(text):
    """
    Method to clean the text by removing special characters, numbers, and converting to lowercase.
    """
    text = text.lower() # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)  # Remove specific punctuations
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove non-ASCII and emojis
    text = re.sub(r'([.,!?-])', r' \1 ', text)  # Add space around specific punctuations
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Limit character repetition (e.g., "soooo" -> "soo")
    return text.strip() 

def preprocess_pipeline(text):
    """
    Method to preprocess text data by tokenize, remove stopwords, and lemmatize.
    """
    tokens = tokenize(text)
    lemmatizer = WordNetLemmatizer()
    processed = [lemmatizer.lemmatize(t) for t in tokens if t not in english_stopwords]
    return ' '.join(processed)

def predict_sentiment(review, model, tokenizer):
    """
    Method for predicting sentiment of a review using a trained model and tokenizer.
    """
    review = clean_pipeline(review)
    review = preprocess_pipeline(review)
    # tokenize and pad the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)[0]
     # Convert padded_sequence to a PyTorch tensor and move to the device
    padded_sequence = torch.tensor(padded_sequence, dtype=torch.long).unsqueeze(0).to(device)
    prediction = model(padded_sequence)

    # sentiment = 'Positive 😄' if prediction > 0.5 else 'Negative'
    if prediction > 0.9:
        sentiment = 'Positive 😍'
    elif prediction > 0.7:
        sentiment = 'Positive 😄'
    elif prediction > 0.5:
        sentiment = 'Positive 😀'
    elif prediction > 0.3:
        sentiment = 'Negative 😤'
    elif prediction > 0.1:
        sentiment = 'Negative 😨'
    else:
        sentiment = 'Negative 😫'
    return sentiment, prediction.item()

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=400, n_layers=2, dropout=0.2):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) # embedding layer - input into vector representation
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True) # LSTM layer
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output
        self.sigmoid = nn.Sigmoid() # Sigmoid layer for converting binary classification

    def forward(self, x):
        x = x.long() # converting feature to long
        x = self.embedding(x) # map input to vector
        o, _ =  self.lstm(x) # pass forward to lstm
        o = o[:, -1, :] # get last sequence output
        o = self.dropout(o) # apply dropout and fully connected layer
        o = self.fc(o)
        o = self.sigmoid(o) # sigmoid Layer

        return o

with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentModel(vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)
model.load_state_dict(torch.load('sentiment_model.pth', map_location=device))
model.to(device)
model.eval()

def display_reviews(reviews, model, tokenizer):
    """
    Creates HTML Cards and displays the sentiment prediction results
    """
    html_content = """
    <style>
        .review-container {
            width: 100%;
            max-height: 400px;  /* Adjustable max height */
            overflow-y: auto;   /* Enable vertical scrolling */
            margin: 20px 0;
        }
        .review-card {
            border: 1px solid #ccc;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            padding: 15px;
            font-family: Arial, sans-serif;
            background: #f9f9f9;
            width: 100%;
            box-sizing: border-box;
        }
        .review-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .review-text {
            font-size: 16px;
            margin-bottom: 10px;
            color: #555;
            word-wrap: break-word;
        }
        .predicted {
            font-size: 14px;
            margin-top: 10px;
            font-weight: bold;
        }
        .positive {
            color: #2ecc71;
        }
        .negative {
            color: #e74c3c;
        }
    </style>
    """

    html_content += '<div class="review-container">'
    for i, review in enumerate(reviews, start=1):
        sentiment, score = predict_sentiment(review["review"], model, tokenizer)
        sentiment_class = "positive" if sentiment == "positive" else "negative"

        rating = review.get("rating", "N/A")

        html_content += f"""
        <div class="review-card">
            <div class="review-title">Given Review :</div>
            <div class="review-text">{review['review']}</div>
            <div class="predicted {sentiment_class}">Predicted Sentiment: {sentiment.capitalize()} | Score: {score:.4f}</div>
        </div>
        """
    html_content += '</div>'

    return html_content


def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            color: #F04614;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px #cccccc;
        }
        .stTextArea textarea {
            background-color: #f8f9fa;
            border: 2px solid #F04614;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #F04614;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #d43c12;
            transform: scale(1.05);
        }
        .clear-btn>button {
            background-color: #666;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            transition: all 0.3s;
        }
        .clear-btn>button:hover {
            background-color: #444;
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="main-title">🎬 IMDB Sentiment Analyzer</div>', unsafe_allow_html=True)

    # Define default review
    DEFAULT_REVIEW = "I can't agree with the negative reviews on here, sure it's not a deep artist masterpiece, but it is adorable and fun! I think the cast is all super cute and they did a really great job acting their parts. The characters I think r fleshed out and endearing."

    # Initialize session state for text only if not already set
    if 'review_text' not in st.session_state:
        st.session_state.review_text = DEFAULT_REVIEW

    # Text area for review input
    review_input = st.text_area(
        "Enter your movie review here:",
        value=st.session_state.review_text,
        height=200,
        key="review_input"
    )

    # Create two columns for buttons
    col1, col2 = st.columns(2)

    with col1:
        # Analyze button
        analyze_clicked = st.button("Analyze Sentiment", key="analyze")

    with col2:
        # Clear button
        if st.button("Clear", key="clear", help="Clear the text box", type="primary"):
            st.session_state.review_text = ""
            st.rerun()

    if analyze_clicked:
        if review_input.strip():
            st.session_state.review_text = review_input
            review_data = [{
                "review": review_input,
                "label": "Unknown",
                "rating": "N/A"
            }]
            html_result = display_reviews(review_data, model, tokenizer)
            components.html(html_result, height=400)
        else:
            st.warning("Please enter a review before analyzing!")
    # Add some footer styling
    st.markdown("""
    <hr style='border: 1px solid #F04614; margin-top: 40px;'>
    <p style='text-align: center; color: #666; font-size: 14px;'>
        Created with Streamlit ❤
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()