import os
import fitz  # PyMuPDF
import csv

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_data_db")
collection = client.create_collection(name="document_collection")

folder_path = './pdf_folder'

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        doc = fitz.open(pdf_path)  # Open the PDF
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()  # Extract text from the page
        return text
    except Exception as e:
        # Log the error and return an empty string or a message
        print(f"Error processing {pdf_path}: {e}")
        return ""

# Function to extract text from a CSV file
def extract_text_from_csv(csv_path):
    text = ""
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            text += ' '.join(row) + '\n'  # Join CSV row elements with space
    return text

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to store content from different files into ChromaDB
def store_file_content_in_chromadb():
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process based on file extension
        if filename.endswith(".pdf"):
            print(f"Processing PDF: {filename}")
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".csv"):
            print(f"Processing CSV: {filename}")
            text = extract_text_from_csv(file_path)
        elif filename.endswith(".txt"):
            print(f"Processing TXT: {filename}")
            text = extract_text_from_txt(file_path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue  # Skip unsupported file types

        # Add the document content to ChromaDB
        collection.add(
            documents=[text],
            metadatas=[{"filename": filename}],
            ids=[filename],  # Using filename as a unique identifier
        )
        print(f"Stored content from {filename} in ChromaDB")

# Call the function to process and store files
store_file_content_in_chromadb()

