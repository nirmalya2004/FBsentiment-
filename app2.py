from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import matplotlib.pyplot as plt

# Set up the Streamlit page configuration
st.set_page_config(page_title="Facebook Sentiment Analysis Dashboard", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f5;
        color: #333;
    }
    .header {
        font-size: 40px;
        text-align: center;
        color: #007bff;
        margin-top: 20px;
    }
    .expander {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .success {
        color: #28a745;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='header'>Facebook Sentiment Analysis</h1>", unsafe_allow_html=True)

# Function to map polarity score to emotions
def get_emotion(polarity):
    if polarity > 0.5:
        return "Very Positive"
    elif 0 < polarity <= 0.5:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    elif -0.5 <= polarity < 0:
        return "Negative"
    else:
        return "Very Negative"

# Analyze Text Input
with st.expander('Analyze Text', expanded=True):
    text = st.text_input('Enter text for sentiment analysis:')
    if text:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        emotion = get_emotion(polarity)
        st.success(f'Polarity: {polarity} ({emotion})', icon="✅")
        st.success(f'Subjectivity: {round(blob.sentiment.subjectivity, 2)}', icon="✅")

# Analyze CSV Upload
with st.expander('Analyze CSV', expanded=True):
    upl = st.file_uploader('Upload your CSV file here:', type=['csv', 'xlsx'])

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        try:
            # Load CSV or Excel file
            if upl.name.endswith('.csv'):
                df = pd.read_csv(upl)
            else:
                df = pd.read_excel(upl)

            if 'FBPost' not in df.columns:
                st.error("The uploaded file must contain a 'FBPost' column.")
            else:
                df['score'] = df['FBPost'].apply(score)
                df['analysis'] = df['score'].apply(analyze)
                df['emotion'] = df['score'].apply(get_emotion)

                # Display the first 10 results
                st.write(df[['FBPost', 'score', 'emotion']].head(10))

                # Create a bar chart for sentiment distribution
                sentiment_counts = df['emotion'].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue', 'orange', 'purple'])
                ax.set_title('Sentiment Distribution')
                ax.set_ylabel('Count')
                ax.set_xlabel('Emotion')

                # Adding labels to bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='bottom')

                st.pyplot(fig)

                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='sentiment_analysis.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
