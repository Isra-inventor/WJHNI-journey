from flask import Flask, request, render_template, redirect, url_for
import os
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
from lucknowllm import UnstructuredDataLoader, split_into_segments
from lucknowllm import GeminiModel
import google.generativeai as gen_ai

app = Flask(__name__)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to calculate cosine similarity between two sets of vectors

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

# Function to scrape content from a given URL
def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        print(f"An error occurred while scraping the website: {e}")
        return None

# Load data from a text file
file_path = "./Data.txt"
with open(file_path, "r") as file:
    external_data = file.readlines()
chunks = split_into_segments(external_data[0])
embedded_data = model.encode(chunks)

# List of URLs to scrape
urls = [
    "https://www.esm-tlemcen.dz/admission-et-inscription/",
    "https://www.univ-bechar.dz/Ar/index.php/2024/07/17/progres-paiement-inscription-transport-hubergement/#:~:text=%D8%AF%D9%81%D8%B9%20%D8%AD%D9%82%D9%88%D9%82%20%D8%A7%D9%84%D8%A7%D9%8A%D9%88%D8%A7%D8%A1%20(%20%D8%A7%D9%84%D8%A7%D9%82%D8%A7%D9%85%D8%A9%20)%202024,%D8%A7%D9%84%D8%AA%D8%B3%D8%AC%D9%8A%D9%84%20%D8%8C%20%D8%AF%D9%88%D9%86%20%D8%AA%D9%86%D9%82%D9%84%20%D8%A7%D9%84%D9%89%20%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D8%A9.",
    "https://shs.cu-tipaza.dz/ar/%D8%A7%D9%84%D9%85%D9%86%D8%AD%D8%A9-%D9%88-%D8%A7%D9%84%D8%A7%D9%8A%D9%88%D8%A7%D8%A1-%D9%88-%D8%A7%D9%84%D8%A5%D9%82%D8%A7%D9%85%D8%A9-%D9%88-%D8%A7%D9%84%D9%86%D9%82%D9%84-%D8%A7%D9%84%D8%AC%D8%A7/",
    "https://www.ens-ouargla.dz/%D8%B1%D8%B2%D9%86%D8%A7%D9%85%D8%A9-%D8%A7%D9%84%D8%AA%D8%B3%D8%AC%D9%8A%D9%84%D8%A7%D8%AA-%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D9%8A%D8%A9-%D9%84%D9%84%D8%B3%D9%86%D8%A9-%D8%A7%D9%84%D8%AC%D8%A7/",
    "https://www.echoroukonline.com/%D9%83%D9%84-%D8%AA%D9%81%D8%A7%D8%B5%D9%8A%D9%84-%D8%A7%D9%84%D8%AA%D8%B3%D8%AC%D9%8A%D9%84%D8%A7%D8%AA-%D9%88%D8%A7%D9%84%D8%AA%D9%88%D8%AC%D9%8A%D9%87-%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D9%8A",
    "https://www.orientation-dz.com/2024/08/28/%d9%85%d9%88%d8%b9%d8%af-%d8%a7%d9%86%d8%b7%d9%84%d8%a7%d9%82-%d8%a7%d9%84%d8%af%d8%b1%d9%88%d8%b3-%d9%84%d9%84%d8%b3%d9%86%d8%a9-%d8%a7%d9%84%d8%ac%d8%a7%d9%85%d8%b9%d9%8a%d8%a9-2024-2025/",
    "https://www.orientation-dz.com/2024/08/18/%d9%83%d9%8a%d9%81%d9%8a%d8%a9-%d8%a7%d9%84%d8%aa%d8%ad%d9%88%d9%8a%d9%84-%d8%a7%d9%84%d8%ac%d8%a7%d9%85%d8%b9%d9%8a-%d9%84%d9%84%d8%ac%d8%af%d8%af-%d8%a8%d9%83%d8%a7%d9%84%d9%88%d8%b1%d9%8a%d8%a7-202/",
    "https://horizons-edu.com/blog/%D8%A7%D8%AE%D8%AA%D9%8A%D8%A7%D8%B1-%D8%A7%D9%84%D8%AA%D8%AE%D8%B5%D8%B5-%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D9%8A",
    "https://www.ency-education.com/specialities-information.html",
    "https://www.mstaml.com/dz/post/%D8%A7%D9%84%D8%AC%D8%B2%D8%A7%D8%A6%D8%B1/%D9%83%D9%85%D8%A7%D9%84%D9%8A%D8%A7%D8%AA-%D9%85%D9%86%D9%88%D8%B9%D8%A7%D8%AA/%D9%83%D9%8A%D9%81-%D8%AA%D8%AE%D8%AA%D8%A7%D8%B1-%D8%A7%D9%84%D8%AA%D8%AE%D8%B5%D8%B5-%D8%A7%D9%84%D9%85%D9%86%D8%A7%D8%B3%D8%A8-%D9%84%D9%83-%D9%81%D9%8A-%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D8%A9-%D9%81%D9%8A-%D8%A7%D9%84%D8%AC%D8%B2%D8%A7%D8%A6%D8%B1?id=266364&location=4327&type=8",
    "https://al-ain.com/article/algerian-universities",
    "https://www.a-onec.com/2023/07/best-university-majors.html",
    "https://www.elkhabar.com/press/article/246378/5-%D8%AA%D8%B7%D8%A8%D9%8A%D9%82%D8%A7%D8%AA-%D8%AA%D8%AD%D8%AA-%D8%AA%D8%B5%D8%B1%D9%81-%D8%A7%D9%84%D9%86%D8%A7%D8%AC%D8%AD%D9%8A%D9%86-%D9%81%D9%8A-%D8%A7%D9%84%D8%A8%D9%83%D8%A7%D9%84%D9%88%D8%B1%D9%8A%D8%A7/",
    "https://www.amar-info1.com/2024/06/Strong-university-majors.html",
    "https://www.elkhabar.com/press/article/247114/%D9%87%D8%B0%D9%87-%D9%87%D9%8A-%D9%85%D8%B9%D8%AF%D9%84%D8%A7%D8%AA-%D8%A7%D9%84%D8%AA%D8%B3%D8%AC%D9%8A%D9%84-%D8%A7%D9%84%D8%AC%D8%A7%D9%85%D8%B9%D9%8A-%D9%88%D8%A7%D9%84%D8%AA%D8%AE%D8%B5%D8%B5%D8%A7%D8%AA/",
]

# Scrape content from each website and integrate it into the pipeline
for url in urls:
    web_content = scrape_website(url)
    print(web_content)
    if web_content:
        # Segment and encode the scraped content
        web_chunks = split_into_segments(web_content)
        embedded_web_data = model.encode(web_chunks)
        # Combine file data embeddings and web data embeddings
        embedded_data = np.concatenate((embedded_data, embedded_web_data), axis=0)
        chunks.extend(web_chunks)
        
gen_ai_api_key = "AIzaSyDU3GInlQk8GBl6SSb10Li6cs2beLI_8Dw"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tool', methods=['GET', 'POST'])
def tool():
    gen_ai.configure(api_key=gen_ai_api_key)
    generation_config = {
    "temperature": 0,               
    "top_p": 0.9,                     
    "top_k": 50,                      
    "max_output_tokens": 1000,        
    }
    Gemini = gen_ai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    if request.method == 'POST':
        # Get the input data from the form
        queries = request.form['user_input']
        embedded_queries = model.encode(queries)
        #Translating query depending on its content to fit information in the Data set
        prompt = (
        f"if the text is asking about the minimum scores to get in a major translate it to French"
        f"if the text is asking anything else, translate it to Arabic"
        f"output ONLY the translated text without anything else"
        f"Query: {queries}"
        )
        try:
            output = Gemini.generate_content(prompt)
            queries=[output.text]
        except Exception as e:
            print(f"An error occurred while generating content: {e}")
        embedded_queries = model.encode(queries)
        generation_config = {
            "temperature": 0,               # Adjusted for some randomness
            "top_p": 0.9,                     # Use top-p sampling with a cumulative probability of 0.9
            "top_k": 50,                      # Consider the top 50 tokens in each step
            "max_output_tokens": 1000,        # Adjusted to a more reasonable maximum output length
        }

        Gemini = gen_ai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config)
        # Retrieve the most similar chunks and generate answers
        for i, query_vec in enumerate(embedded_queries):
            # Compute similarities between the query and the embedded data
            similarities = cosine_similarity(query_vec[np.newaxis, :], embedded_data).flatten()
            # Get top 5 indices based on similarities
            top_indices = np.argsort(similarities)[::-1][:80]
            top_doct = [chunks[index] for index in top_indices]
            # Compute cosine similarities between each query and all documents
            embedded_doct = model.encode(top_doct)
            similarity_matrix = cosine_similarity(embedded_queries, embedded_doct)
            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
            query_index, doct_index = max_sim_idx
            most_similar = top_doct[doct_index]
            top_n = 10
            l=[]
            for i, query in enumerate(queries):
            # Get indices of the top 3 most similar documents
            top_indices = np.argsort(similarity_matrix[i])[::-1][:top_n]
            for j in range(top_n):
                nearest_index = top_indices[j]
                nearest_document = top_doct[nearest_index]
                similarity_score = similarity_matrix[i][nearest_index] 
                l.append(nearest_document)
            # Construct the prompt for the generative model
            argumented_prompt = (
                f"You are an expert question answering system about the Algerian Baccalaureate"
                f"Answer the question in Arabic if asked in Arabic. If asked in French, answer in French, Always give a specified accurate answer"
                f"When asked to help choose a major, ask about the user's interests and experiences if not provided"
                f"When asked if the user will be accepted in a certain major ask about their score/grade and compare to the minimum score required"
                f"Make sure to know that the Superior schools always require a very high score, differentite between those and normal majors "
                f"Query: {queries[i]} Contexts: {l}"
            )
    
            # Generate content using the Gemini model
            try:
                model_output = Gemini.generate_content(argumented_prompt)
                result=model_output.text
            except Exception as e:
                print(f"An error occurred while generating content: {e}")
        return render_template('tool.html', result=result)
    return render_template('tool.html')
if __name__ == "__main__":
    app.run(debug=True)
