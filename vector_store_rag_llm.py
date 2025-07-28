import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer  # For embeddings
from groq import Groq
import traceback

load_dotenv()

# Initialize ChromaDB client
#client = chromadb.PersistentClient(path="./data_db")
#client = chromadb.PersistentClient(path=r"C:\Users\Md Shahabuddin\Desktop\test_folder\data_db")
client = chromadb.PersistentClient(path=r"./chroma_data_db")
collection = client.get_or_create_collection(name="document_collection")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to generate embedding for a query
def generate_embedding(text):
    embedding = embedding_model.encode(text, show_progress_bar=False)
    return embedding.tolist() 


def search_chromadb_with_embedding(query, n_results=2):
    query_embedding = generate_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    return results['documents']

# Function to read the prompt template from a file
def read_prompt_template():
    with open("prompt_template.txt", "r") as file:
        template = file.read()
    return template

def build_prompt(query, context):
    prompt_template = read_prompt_template()
    
    if not context:
        context = "<No relevant context found>"
    
    # Replace placeholders with actual values in the prompt template
    prompt = prompt_template.format(query=query, context=context)
    
    return prompt


def get_answer_from_groq(prompt):
    try:
        # Send the prompt to the model and get a response
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content.strip()

        return answer
    
    except AttributeError as e:
        traceback.print_exc()
        return "Error: Could not extract the answer due to missing attribute."
    
    except Exception as e:
        traceback.print_exc()
        return "Error: Something went wrong while generating the answer."



# Function to run the query and generate an answer
def run_query(query):
    relevant_documents = search_chromadb_with_embedding(query)

    context = "\n".join(
        doc[0] for doc in relevant_documents
        if isinstance(doc, list) and len(doc) > 0
    )
    
    prompt = build_prompt(query, context)
    answer = get_answer_from_groq(prompt)
    
    return answer
