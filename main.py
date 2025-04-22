import os
import json
import glob
import hashlib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, write_index, read_index
from nltk.tokenize import sent_tokenize
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from cachetools import LRUCache
from dotenv import load_dotenv

# Download NLTK data
nltk.download('punkt')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Load configuration
def load_config(config_file="config.json"):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found")

config = load_config()
EMBEDDING_MODEL = config.get('embedding_model', 'all-MiniLM-L6-v2')
LLM_MODEL_REFORMULATE = config.get('llm_model_reformulate', 'llama3-8b-8192')
LLM_MODEL_ANSWER = config.get('llm_model_answer', 'gemma2-9b-it')

# Initialize components
embedder = SentenceTransformer(EMBEDDING_MODEL)
groq_client = Groq(api_key=GROQ_API_KEY)
query_cache = LRUCache(maxsize=100)  # Cache for query embeddings
intent_cache = LRUCache(maxsize=100)  # Cache for intent classification
reformulation_cache = LRUCache(maxsize=100)  # Cache for query reformulation
answer_cache = LRUCache(maxsize=100)  # Cache for answer generation

# Chat memory
chat_history = []

# Load pre-trained intent classifier and vectorizer
try:
    with open("intent_classifier.pkl", "rb") as f:
        intent_classifier = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("Loaded pre-trained intent classifier and TF-IDF vectorizer")
except FileNotFoundError as e:
    print(f"Failed to load classifier or vectorizer: {e}")
    raise

# Load JSON data
def load_json_data(data_dir="./data"):
    all_data = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                for entry in data:
                    if not all(key in entry for key in ['question', 'answer', 'intent']):
                        continue
                    entry['file'] = os.path.basename(file)
                    entry['id'] = hashlib.md5(entry['question'].encode()).hexdigest()[:8]
                    all_data.append(entry)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON in {file}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid JSON data loaded from {data_dir}")
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} JSON entries")
    return df

# Chunk data
def chunk_data(df, max_chunk_size=500):
    chunks = []
    for _, row in df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        sentences = sent_tokenize(text)
        current_chunk = ""
        chunk_id = 0
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"{row['id']}_{chunk_id}",
                        'text': current_chunk.strip(),
                        'intent': row['intent'],
                        'file': row['file'],
                        'question_id': row['id']
                    })
                    chunk_id += 1
                    current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append({
                'id': f"{row['id']}_{chunk_id}",
                'text': current_chunk.strip(),
                'intent': row['intent'],
                'file': row['file'],
                'question_id': row['id']
            })
    chunks_df = pd.DataFrame(chunks)
    print(f"Created {len(chunks_df)} chunks")
    return chunks_df

# Load into FAISS
def load_to_faiss(chunks_df, index_file="faiss_index.bin"):
    texts = chunks_df['text'].tolist()
    if os.path.exists(index_file):
        index = read_index(index_file)
        embeddings = np.load('embeddings.npy')
    else:
        embeddings = embedder.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = IndexFlatL2(dimension)
        index.add(embeddings)
        write_index(index, index_file)
        np.save('embeddings.npy', embeddings)
    return index, texts, embeddings, chunks_df

# TF-IDF retrieval
def tfidf_retrieval(query, texts, top_k=50):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        scores = (tfidf_matrix * query_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return top_indices
    except Exception as e:
        print(f"TF-IDF retrieval failed: {e}")
        return []

# Semantic search with FAISS
def faiss_search(query, index, embeddings, texts, chunks_df, k=20):
    try:
        query_key = hashlib.md5(query.encode()).hexdigest()
        if query_key in query_cache:
            query_emb = query_cache[query_key]
        else:
            query_emb = embedder.encode([query])[0]
            query_cache[query_key] = query_emb
        distances, indices = index.search(np.array([query_emb]), k)
        results = [
            {
                'text': texts[idx],
                'metadata': chunks_df.iloc[idx][['id', 'intent', 'file', 'question_id']].to_dict(),
                'score': 1 / (1 + dist)
            }
            for idx, dist in zip(indices[0], distances[0])
        ]
        return results
    except Exception as e:
        print(f"FAISS search failed: {e}")
        return []

# Reranking
def rerank_chunks(query, chunks, top_n=5):
    try:
        query_key = hashlib.md5(query.encode()).hexdigest()
        if query_key in query_cache:
            query_emb = query_cache[query_key]
        else:
            query_emb = embedder.encode([query])[0]
            query_cache[query_key] = query_emb
        chunk_texts = [c['text'] for c in chunks]
        chunk_embs = embedder.encode(chunk_texts)
        scores = [
            np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
            for chunk_emb in chunk_embs
        ]
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [chunk for chunk, _ in ranked], [score for _, score in ranked]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return [], []

# Query reformulation using LLM with caching
def reformulate_query(query):
    query_key = hashlib.md5(query.encode()).hexdigest()
    if query_key in reformulation_cache:
        print(f"Using cached reformulated query for: {query}")
        return reformulation_cache[query_key]
    
    try:
        prompt = f"Reformulate the following query to be concise and clear for a university FAQ system:\nQuery: {query}\nReformulated:"
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_REFORMULATE,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        reformulated = response.choices[0].message.content.strip()
        reformulation_cache[query_key] = reformulated
        return reformulated
    except Exception as e:
        print(f"Query reformulation failed: {e}")
        return query

# Intent classification using pre-trained ML model with caching
def classify_intent(query):
    query_key = hashlib.md5(query.encode()).hexdigest()
    if query_key in intent_cache:
        print(f"Using cached intent for: {query}")
        return intent_cache[query_key]
    
    try:
        query_tfidf = tfidf_vectorizer.transform([query])
        intent = intent_classifier.predict(query_tfidf)[0]
        intent_cache[query_key] = intent
        return intent
    except Exception as e:
        print(f"Intent classification failed: {e}")
        return "unknown"

# Answer generation using LLM with caching
def generate_answer(prompt):
    prompt_key = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_key in answer_cache:
        print(f"Using cached answer for prompt")
        return answer_cache[prompt_key]
    
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_ANSWER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        answer_cache[prompt_key] = answer
        return answer
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return None

# RAG pipeline
def rag_pipeline(query, faiss_index, texts, embeddings, chunks_df, max_attempts=2):
    global chat_history
    print(f"Processing query: {query}")

    attempt = 0
    reformed_query = query
    while attempt < max_attempts:
        reformed_query = reformulate_query(query) if attempt > 0 else query
        print(f"Reformed query: {reformed_query}")

        intent = classify_intent(reformed_query)
        print(f"Classified intent: {intent}")

        tfidf_indices = tfidf_retrieval(reformed_query, texts)
        print(f"TF-IDF retrieved {len(tfidf_indices)} indices")

        search_results = faiss_search(reformed_query, faiss_index, embeddings, texts, chunks_df, k=20)
        chunks = [res for res in search_results]
        print(f"Found {len(chunks)} chunks in FAISS search")

        ranked_chunks, scores = rerank_chunks(reformed_query, chunks)
        if not ranked_chunks or (scores and max(scores) < 0.6):
            print(f"Low confidence or no chunks, attempt {attempt + 1}")
            attempt += 1
            continue

        context = "\n\n".join([chunk['text'] for chunk in ranked_chunks])
        history_context = "\n".join(
            [f"User: {h['query']}\nBot: {h['answer']}" for h in chat_history[-3:]]
        )
        prompt = f"""
        Using the following context and conversation history, answer the query concisely. Cite the question ID.
        Context:
        {context}
        Conversation History:
        {history_context}
        Query: {reformed_query}
        Answer:
        """
        answer = generate_answer(prompt)
        if not answer:
            attempt += 1
            continue

        sources = [chunk['metadata']['question_id'] for chunk in ranked_chunks]
        evaluation = {'faithfulness': 0.0, 'answer_relevancy': 0.0}  # Placeholder

        chat_history.append({
            'query': query,
            'answer': answer,
            'reformed_query': reformed_query,
            'sources': sources,
            'confidence': max(scores)
        })

        print(f"Query answered with confidence: {max(scores)}")
        return {
            'answer': answer,
            'sources': sources,
            'confidence': max(scores),
            'reformed_query': reformed_query,
            'evaluation': evaluation
        }
    
    print(f"Failed to find relevant answer after {max_attempts} attempts")
    chat_history.append({
        'query': query,
        'answer': "Sorry, I couldn't find a relevant answer.",
        'reformed_query': reformed_query,
        'sources': [],
        'confidence': 0.0
    })
    return {
        'answer': "Sorry, I couldn't find a relevant answer.",
        'sources': [],
        'confidence': 0.0,
        'reformed_query': reformed_query,
        'evaluation': {}
    }

# Main CLI interface
def main():
    print("SRM Chatbot Ready! Type 'exit' to quit.")
    
    try:
        df = load_json_data()
        chunks_df = chunk_data(df)
        faiss_index, texts, embeddings, chunks_df = load_to_faiss(chunks_df)
    except Exception as e:
        print(f"Error: Failed to load data: {e}")
        return

    while True:
        query = input("Your question: ").strip()
        if query.lower() == 'exit':
            break
        
        result = rag_pipeline(query, faiss_index, texts, embeddings, chunks_df)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reformed Query: {result.get('reformed_query', 'N/A')}")
        print(f"Evaluation: {result['evaluation']}\n")
        
        print("Chat History:")
        for i, chat in enumerate(chat_history):
            print(f"Q{i+1}: {chat['query']}")
            print(f"A{i+1}: {chat['answer']}")
            print("---")

if __name__ == "__main__":
    main()