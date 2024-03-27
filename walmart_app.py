import streamlit as st
from sentiment_analysis import walmart_reviews, predicting_pipeline_textblob, plot_rating_distribution, scrape_walmart_reviews
import pandas as pd

# Set the page configuration to use a wide layout
st.set_page_config(layout="wide")

def main():
    # Title and subtitle of the app
    st.title("Walmart Product Sentiment Analyzer")
    st.subheader("Electronic Department Product Analysis")

    # Instructions and disclaimer
    st.markdown(
        """
        This analysis is focused exclusively on products from Walmart's Electronic Department. 
        Use this app for educational purposes only, and ensure scraping is performed in compliance 
        with Walmart's `robots.txt`.
        
        To find a product code, visit [Walmart's Electronics Department](https://www.walmart.com/cp/electronics/3944?povid=GlobalNav_rWeb_Electronics_Electronics_ShopAll).
        """
    )

    # Fetch the list of products from your data source
    raw_url = 'https://raw.githubusercontent.com/NetanelFarhi/Walmart-Data/main/walmart_reviews.csv'
    df_all_products = pd.read_csv(raw_url, encoding='ISO-8859-1')
    
    # Remove "Back to" from product names and filter for electronics department products
    electronics_department = 'Electronics'  # Replace with the actual department name or ID if different
    df_all_products['Product Name'] = df_all_products['Product Name'].str.replace('Back to', '', regex=False)
    df_electronics = df_all_products[df_all_products['Department'] == electronics_department]
    products_list = df_electronics['Product Name'].unique().tolist()

    # Sidebar for selecting a product and initiating analysis or scraping
    with st.sidebar:
        st.write("## Select a Product from the Electronics Department")
        selected_product_name = st.selectbox("Choose a product", products_list)
        analyze_button = st.button("Analyze Sentiment")
        # Sidebar option for scraping reviews
        st.write("## Scrape Reviews")
        st.markdown(
            """
            If you want to scrape reviews for a specific product, first ensure that you are 
            compliant with Walmart's policies. You can locate the product ID from the URL of 
            the product page on Walmart's Electronics Department section.
            """
        )
        product_id = st.text_input("Enter Product ID to scrape", key="product_id_input")
        scrape_button = st.button("Start Scraping")

    # Main area for sentiment analysis
    if analyze_button and selected_product_name:
        with st.spinner("Fetching and analyzing reviews... This may take a moment."):
            try:
                product_code = df_electronics[df_electronics['Product Name'] == selected_product_name]['Product ID'].iloc[0]
                df_reviews = walmart_reviews(product_code)
                fig_sentiment = predicting_pipeline_textblob(df_reviews)
                fig_rating = plot_rating_distribution(df_reviews)
                
                st.success("Sentiment Analysis Completed!")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_sentiment)  # Display sentiment analysis
                with col2:
                    st.pyplot(fig_rating)  # Display rating distribution

            except Exception as e:
                st.error(f"Failed to analyze sentiment: {str(e)}")

        st.write("### Sample Reviews")
        sample_reviews = df_reviews['Review Text'].sample(n=5).tolist()
        for review in sample_reviews:
            st.info(review)

    # Scrape reviews based on product ID input
    if scrape_button and product_id:
        with st.spinner(f"Scraping reviews for Product ID: {product_id}..."):
            try:
                scraped_data = scrape_walmart_reviews([product_id])
                if not scraped_data.empty:
                    st.success(f"Scraping completed! Found {len(scraped_data)} reviews.")
                    st.write(scraped_data)  # Display the scraped data in the app
                else:
                    st.warning("No reviews were found for the entered Product ID.")
            except Exception as e:
                st.error(f"An error occurred during scraping: {e}")

if __name__ == "__main__":
    main()
