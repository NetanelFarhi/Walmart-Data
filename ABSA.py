import streamlit as st
from sentiment_analysis import scrape_walmart_reviews, predicting_pipeline_textblob,reviews_scraper
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# Your reviews_scraper function with necessary modifications here
st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.title("Walmart Product Sentiment Analyzer")
    st.subheader("Discover what customers think about Walmart products!")

    raw_url = 'https://raw.githubusercontent.com/NetanelFarhi/Walmart-Data/main/walmart_reviews.csv'
    df_all_products = pd.read_csv(raw_url, encoding='ISO-8859-1')
    df_all_products=df_all_products[df_all_products['Department'] == "Electronics"]
    df_all_products['Product Name'] = df_all_products['Product Name'].str.replace("Back to ", "", regex=False)
    
    
    products_list = df_all_products['Product Name'].unique().tolist()

    with st.sidebar:
        st.write("## Select a Product")
        selected_product_name = st.selectbox("Choose a product", products_list)
        analyze_button = st.button("Analyze Sentiment")
        
        # New UI components for scraping by product ID
        st.write("## Or Enter a Product ID to Scrape Reviews")
        scrape_product_button = st.button("Scrape Product")
        product_id = st.text_input("Enter the Walmart Product ID here")

    if analyze_button:
        with st.spinner("Fetching and analyzing reviews... This may take a moment."):
            try:
                # Find the product code corresponding to the selected product name
                product_code = df_all_products[df_all_products['Product Name'] == selected_product_name]['Product ID'].iloc[0]
                # Replace with actual scraping and analysis functions
                df_reviews = scrape_walmart_reviews(product_code)
                fig = predicting_pipeline_textblob(df_reviews)
                
                st.success("Sentiment Analysis Completed!")
                st.pyplot(fig)  # Pass the fig object explicitly to st.pyplot()
            except Exception as e:
                st.error(f"Failed to analyze sentiment: {str(e)}")

        st.write("### Sample Reviews")
        # Display some sample reviews for the selected product
        sample_reviews = df_reviews['Review Text'].sample(n=5).tolist()
        for review in sample_reviews:
            st.info(review)
        

    if scrape_product_button and product_id:
      with st.spinner("Scraping reviews... This may take a moment."):
        try:
            # Call your modified reviews_scraper function with the input product_id
            df_reviews = reviews_scraper(product_id)
            if not df_reviews.empty:
                st.success(f"Scraped {len(df_reviews)} reviews!")
                
                # Perform sentiment analysis on the scraped reviews
                with st.spinner("Analyzing sentiment..."):
                    fig = predicting_pipeline_textblob(df_reviews)
                    st.success("Sentiment Analysis Completed!")
                    st.pyplot(fig)  # Display the sentiment analysis results
                
                # Optionally, display some sample reviews
                st.write("### Sample Reviews")
                sample_reviews = df_reviews['Review Text'].sample(n=5).tolist()
                for review in sample_reviews:
                    st.info(review)
            else:
                st.info("No reviews found for this product.")
        except Exception as e:
            st.error(f"Failed to scrape or analyze reviews: {str(e)}")


if __name__ == "__main__":
    main()
