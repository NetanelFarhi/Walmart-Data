import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from lingua import Language, LanguageDetectorBuilder
import inflect
from textblob import TextBlob
from statistics import mean
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_lg")
tfidf_vectorizer = TfidfVectorizer()

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_walmart_reviews(product_id):
    base_url = "https://www.walmart.com/reviews/product/"
    user_agent = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'}
    reviews_data = []

    for product_id in product_id:
        # Initial request to fetch the product name
        product_name_url = f"{base_url}{product_id}"
        response = requests.get(product_name_url, headers=user_agent)
        product_name = "Product name not found"  # Default value
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            a_tag = soup.find('a', class_='w_x7ug f6 dark-gray', href=True)
            if a_tag:
                product_name = a_tag.text  # Extract the product name

        # Proceed to scrape reviews
        for stars in range(1, 6):
            for page in range(1, 3):  # Scrape 2 pages for each star rating
                url = f"{base_url}{product_id}?sort=submission-desc&filter={stars}&page={page}"
                response = requests.get(url, headers=user_agent)
                if response.status_code != 200:
                    continue  # Skip this iteration if the page fails to load
                soup = BeautifulSoup(response.content, 'html.parser')

                review_elements = soup.find_all('div', class_='w_DHV_ pv3 mv0')
                for review_element in review_elements:
                    title_element = review_element.find('h3', class_='w_kV33 w_Sl3f w_mvVb f5 b')
                    title = title_element.text if title_element else 'No Title'
                    
                    text_element = review_element.find('span', class_='tl-m mb3 db-m')
                    text = text_element.text if text_element else 'No Review Text'
                    
                    reviewer_name_element = review_element.find('div', class_='f6 gray pr2 mb2')
                    reviewer_name = reviewer_name_element.text if reviewer_name_element else 'Anonymous'
                    
                    stars_given_element = review_element.find('span', class_='w_iUH7')
                    stars_given = stars_given_element.text if stars_given_element else 'No Rating'
                    
                    review_date_element = review_element.find('div', class_='f7 gray mt1')
                    review_date = review_date_element.text if review_date_element else 'No Date'
                    
                    original_platform_element = review_element.find('div', class_='b ph1 dark gray', attrs={'data-qm-mask': "true"})
                    original_platform = original_platform_element.text if original_platform_element else 'Walmart'
                    
                    reviews_data.append({
                        'Product ID': product_id,
                        'Product Name': product_name,
                        'Title': title,
                        'Review Text': text,
                        'Reviewer Name': reviewer_name,
                        'Original Platform': original_platform,
                        'Stars Given': stars_given,
                        'Review Date': review_date,
                    })
                    
                    

    return pd.DataFrame(reviews_data)


 



def walmart_reviews(product_code):
    # Your scraping function code here
    raw_url = 'https://raw.githubusercontent.com/NetanelFarhi/Walmart-Data/main/walmart_reviews.csv'
    df = pd.read_csv(raw_url, encoding='ISO-8859-1')
    return df.loc[df["Product ID"] == int(product_code)].copy()

import matplotlib.pyplot as plt

def plot_rating_distribution(df):
    # Ensure 'Rating' column is numeric and not string
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Calculate the value counts for each rating and sort by index (rating value)
    rating_counts = df['Rating'].value_counts().sort_index()

    # Create a bar plot for the distribution of ratings
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
    plt.title('Distribution of Rating')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.xticks(rating_counts.index)  # Ensure only integers are used for the x-ticks
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    return plt.gcf()  # Return the figure object to be used with st.pyplot() in Streamlit








#raw_url = 'https://raw.githubusercontent.com/NetanelFarhi/Walmart-Data/main/Walmart%20Reviews%20Data%20Stage2.csv?token=GHSAT0AAAAAACPVTFFN3REYXCOQNSDMQWK2ZPWCZ6Q'
#raw_url = 'https://raw.githubusercontent.com/NetanelFarhi/Walmart-Data/main/walmart_reviews.csv'

# Read the CSV file into a DataFrame
#df = pd.read_csv(raw_url,encoding='ISO-8859-1')

#dm = df.loc[df["Product ID"] == 2311706193].copy()


def drop_duplicates(df):
    subset_columns = ['Product ID', 'Product Name', 'Title', 'Review Text', 'Reviewer Name', 'Original Platform', 'Stars Given']
    if df.duplicated(subset=subset_columns).any():
        df.drop_duplicates(subset=subset_columns, inplace=True)
    return df


def extract_rating(stars_text):
    try:
        rating = int(stars_text.split()[0])
        return rating
    except:
        return None


def detect_language(df, text_column='Review Text', language_column='language'):
   
    # Setup languages and language detector
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Define a function to detect language of a text
    def detect_language(text):
        try:
            # Attempt to detect the language
            language = detector.detect_language_of(text)
            return str(language)
        except Exception as e:
            # In case of any error, return a placeholder or handle the error as needed
            return "Unknown"

    # Apply the detect_language function to each review in the DataFrame
    #df[language_column] = df[text_column].apply(detect_language)
    df.loc[:, language_column] = df[text_column].apply(detect_language)
    return df

def convert_numbers_to_words(text):
    p = inflect.engine()
    # Regular expression pattern to match integers and floats
    pattern = r'\b\d+(?:\.\d+)?\b'
    # Find all numbers in the text using the pattern
    matches = re.findall(pattern, text)
    # Iterate over matches and convert each number into words
    for match in matches:
        # Convert the number into words
        words = p.number_to_words(match)
        # Replace the number with its words in the text
        text = text.replace(match, words)
    return text
nlp = spacy.load("en_core_web_lg")


# extract aspects from a review using both POS tagging and TF-IDF scores
def extract_aspects(review, tfidf_matrix, feature_names, top_n=10):
    aspects_pos = []
    doc = nlp(review)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            aspects_pos.append(token.text)

    review_tfidf = tfidf_vectorizer.transform([preprocess_text(review)])
    review_tfidf = review_tfidf.toarray().flatten()
    top_indices = review_tfidf.argsort()[-top_n:][::-1]
    aspect_terms_idf = [feature_names[idx] for idx in top_indices]

    # Combine aspects from both methods
    combined_aspects = list(set(aspects_pos + aspect_terms_idf))

    return combined_aspects

def preprocess_text(text):
    # Remove non-characters such as digits and symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Handling negations and contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove repeated characters 
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Convert to lowercase
    text = text.lower()

    # remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    text = text.replace('$', 'dollar')

    return text


# Define the function to filter DataFrame based on top 5 most frequent 'Categorized Item'
def filter_top_5_items(df):
    # Count the frequency of each 'Categorized Item' excluding 'other'
    categorized_item_counts = df[df['Aspect Group'] != 'other']['Aspect Group'].value_counts()

    # Get the top 5 most frequent 'Categorized Item'
    top_5 = categorized_item_counts.head(5)
    
    # Filter the dataframe to only include rows with 'Categorized Item' in the top 5
    filtered_df = df[df['Aspect Group'].isin(top_5.index)]
    
    return filtered_df



def Aspect_Group(aspect):
    aspect_groups = {
        'service': ['service', 'support', 'help', 'store', 'worker', 'online', 'website', 'walmart', 'att', 'return', 'warehouse', 'assistance', 'customer', 'care', 'staff', 'employee'],
        'features': ['camera', 'screen', 'processor', 'message', 'feature', 'specification', 'functionality', 'capability'],
        'shipments': ['shipping', 'delivery', 'packaging', 'shipment', 'arrangement', 'dispatch', 'transport', 'delay', 'arrived'],
        'price': ['cost', 'price', 'affordability', 'pricing', 'rate', 'cost-effectiveness', 'value'],
        'battery': ['battery', 'power', 'charge', 'cell', 'energy', 'capacity'],
        'design': ['design', 'appearance', 'look', 'aesthetics', 'style', 'visual', 'form'],
        'performance': ['performance', 'speed', 'efficiency', 'capability', 'execution', 'operation', 'functioning'],
        'compatibility': ['compatibility', 'connectivity', 'integration', 'interoperability', 'suitability', 'adaptability'],
        'ease_of_use': ['ease', 'use', 'user', 'friendly', 'intuitive', 'easy', 'convenience', 'simplicity', 'usability'],
        'reliability': ['reliability', 'durability', 'longevity', 'dependability', 'sturdiness', 'robustness', 'trustworthiness'],
        'sound_quality': ['sound', 'audio', 'speaker', 'acoustics', 'sonic', 'noise', 'soundstage'],
        'display_quality': ['display', 'resolution', 'clarity', 'visuals', 'screen', 'sharpness', 'color', 'brightness'],
        'opinion': ['opinion', 'gift', 'like', 'love', 'great', 'recommend', 'favorite', 'preference', 'choice', 'selection', 'dislike'],
        'feedback': ['feedback', 'review', 'comment', 'critique', 'evaluation', 'assessment', 'input', 'response'],
        'memory_storage': ['memory', 'storage', 'size', 'capacity', 'memory space', 'storage capacity', 'volume'],
        'brand': ['samsung', 'galaxy', 'apple', 'sony', 'lg', 'huawei', 'lenovo', 'dell', 'hp', 'xiaomi', 'asus', 'acer', 'google', 'motorola', 'nokia', 'oppo', 'vivo', 'oneplus', 'htc', 'blackberry', 'realme'],
    }
    
    for category, aspect_list in aspect_groups.items():
        if aspect in aspect_list:
            return category
    return 'other' 


def find_aspect_span(review, aspect, window_size):
    doc = nlp(review)
    aspect_terms = aspect.split()  # Splitting the aspect into terms for matching

    for token in doc:
        if token.text.lower() in [term.lower() for term in aspect_terms]:
            start = max(token.i - window_size, 0)
            end = min(token.i + window_size + 1, len(doc))
            span = doc[start:end]
            return span.text
    return review 
def calculate_polarity_for_windows_textblob(review, aspect, windows):
    polarities = []
    for window in windows:
        segment = find_aspect_span(review, aspect, window)
        analysis = TextBlob(segment)
        polarities.append(analysis.sentiment.polarity)
    
    # Calculate the average polarity
    avg_polarity = mean(polarities) if polarities else 0
   
    # Categorize the sentiment based on the average polarity
    return "Positive" if avg_polarity > 0 else "Negative"
import pandas as pd

def count_aspect_sentiments(df, aspect_col='Aspect Group', sentiment_col='Transformer'):
 
    overall_aspect_sentiment = {}

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        aspect = row[aspect_col]
        sentiment = row[sentiment_col]
        
        # Initialize the aspect in the dictionary if it's not already present
        if aspect not in overall_aspect_sentiment:
            overall_aspect_sentiment[aspect] = {'Positive': 0, 'Negative': 0}
        
        # Update the sentiment count based on the sentiment column's value
        if sentiment == 'Positive':
            overall_aspect_sentiment[aspect]['Positive'] += 1
        elif sentiment == 'Negative':
            overall_aspect_sentiment[aspect]['Negative'] += 1

    # Convert the dictionary to a DataFrame for a tabular view
    sentiment_df = pd.DataFrame.from_dict(overall_aspect_sentiment, orient='index').reset_index()
    sentiment_df.columns = [aspect_col, 'Positive', 'Negative']  # Naming the columns for clarity
    sentiment_df = sentiment_df.set_index(aspect_col)

    return sentiment_df
import matplotlib.pyplot as plt



def plot_aspect_based_sentiment(sentiment_df):
    # Create a new figure and axes object
    fig, ax = plt.subplots()
    
    # Plotting using the axes object
    sentiment_df.plot(kind='bar', color=['green', 'red'], ax=ax)
    
    ax.set_title('Aspect Based Sentiment Analysis')
    ax.set_xlabel('Categorized Aspects')
    ax.set_ylabel('Number of Mentions')
    plt.xticks(rotation=0)  
    plt.tight_layout()  
    plt.legend(title='Sentiment')

    # Return the figure object for further manipulation if needed
    return fig

def predicting_pipeline_textblob(df):    
    #start_time=time.time()
    
    df = drop_duplicates(df)
    
    df=df[df["Review Text"]!="No Review Text"]
    df = df.copy()  
    df.loc[:, 'Rating'] = df['Stars Given'].apply(extract_rating)
    
    df.loc[:, 'Rating'] = df['Stars Given'].apply(extract_rating)
    
    df.loc[:, 'Product Name'] = df['Product Name'].str.replace('Back to', '')
    
    df = detect_language(df)

    df = df.loc[df["language"] == "Language.ENGLISH"].copy()

    
    df['Review Text'] = df['Review Text'].apply(convert_numbers_to_words)

    df['Review Text after'] = df['Review Text'].apply(preprocess_text)
    
        
    # Vectorization with TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review Text after'])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Aspect extraction and categorization
    df["Aspects"] = df["Review Text after"].apply(lambda x: extract_aspects(x, tfidf_matrix, feature_names))
    df = df.explode('Aspects')
    df['Aspect Group'] = df['Aspects'].apply(Aspect_Group)
    df = filter_top_5_items(df)
    windows = [1, 2,3,4]  
    df['TextBlob'] = df.apply(lambda row: calculate_polarity_for_windows_textblob(row['Review Text after'], row['Aspects'], windows), axis=1)

    df = count_aspect_sentiments(df, 'Aspect Group', 'TextBlob')
    plot_aspect_based_sentiment(df)
    #end_time = time.time()  # End time measurement
    #duration = end_time - start_time  # Calculate total duration

#processed_df = predicting_pipeline_textblob(dm)
