import os
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

# Initialize the Groq client
client = Groq(api_key='')

# Configure Google Generative AI with your API key (for embeddings)
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your actual API key

# Function to extract text from a CSV file
def get_csv_content(file):
    try:
        return pd.read_csv(file)  # Read the CSV file into a DataFrame
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

# Function to combine selected columns or all data into text
def combine_columns_to_text(df, selected_columns=None):
    if selected_columns:
        df = df[selected_columns]  # Use only the selected columns
    text = ""
    for _, row in df.iterrows():
        text += " ".join(row.astype(str)) + "\n"  # Combine row values as a single string
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Function to embed text chunks and store in FAISS index
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use your model or preferred one
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Pass the embeddings for generating vectors
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# Function to handle user queries
def handle_query(query, vector_store):
    try:
        search_results = vector_store.similarity_search(query, k=5)  # Perform similarity search
        context = " ".join([result.page_content for result in search_results])  # Combine retrieved text for context
        
        # Create a prompt to send to Groq Cloud
        prompt = f"Answer the following question based on the provided context: {query}\n\nContext: {context}"
        
        # Call Groq Cloud's API for chat completion
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Example model (update as needed)
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,  # Enable streaming to get chunks of response
            stop=None,
        )

        # Collect the response
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return ""

# Streamlit App
st.title("CSV Query App")
st.sidebar.title("Options")

# File upload
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# If multiple files are uploaded
if uploaded_files:
    st.sidebar.write("Select a file to process:")
    file_names = [file.name for file in uploaded_files]  # Get the names of the uploaded files
    selected_file_name = st.sidebar.selectbox("Select a CSV file", file_names)

    # Find the selected file
    selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

    # Load CSV content from the selected file
    df = get_csv_content(selected_file)
    if not df.empty:
        st.write(f"Preview of selected CSV: {selected_file_name}")
        st.dataframe(df)

        # Column selection
        selected_columns = st.sidebar.multiselect("Select columns to use (leave empty for all columns)", df.columns)
        text = combine_columns_to_text(df, selected_columns if selected_columns else None)
        
        # Calendar for filtering date
        if any("date" in col.lower() for col in df.columns):  # Check if any column contains "date"
            date_column = st.sidebar.selectbox("Select a date column for filtering", [col for col in df.columns if "date" in col.lower()])
            selected_date = st.sidebar.date_input("Select a date")
            filtered_df = df[pd.to_datetime(df[date_column]).dt.date == selected_date]
            text = combine_columns_to_text(filtered_df, selected_columns if selected_columns else None)
            st.write(f"Filtered data for {selected_date}:")
            st.dataframe(filtered_df)

        # User query
        query = st.text_input("Enter your query")
        if st.button("Clear"):
            st.experimental_rerun()

        if query:
            st.write("Processing data...")
            text_chunks = split_text_into_chunks(text)  # Split text into chunks
            if text_chunks:
                vector_store = get_vector_store(text_chunks)  # Generate embeddings and store in FAISS index
                if vector_store:
                    st.write("Generating answer...")
                    response = handle_query(query, vector_store)  # Get the response to the query
                    st.success("Answer:")
                    st.write(response)
else:
    st.write("Please upload at least one CSV file.")
