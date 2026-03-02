import streamlit as st
import PyPDF2
import re
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import os
import json
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import ollama
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Page config
st.set_page_config(page_title="GraphRAG Experiment Hub", layout="wide")

# --- CSS for Premium Look ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .highlight {
        color: #e63946;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Resource Loading ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedding_model = load_embedding_model()

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
        except Exception as e:
            self.connected = False
            self.error = str(e)

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_db(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def run_query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

# --- Core Functions ---

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_ollama_response(prompt):
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- RAG Implementations ---

def run_no_rag(resume_text):
    prompt = f"""
    Extract a list of technical skills from the following resume text. 
    Return ONLY a comma-separated list of skills.
    
    Resume Text:
    {resume_text}
    """
    return get_ollama_response(prompt)

def run_vector_rag(resume_text, embedding_model):
    # Simple chunking
    chunks = [resume_text[i:i+500] for i in range(0, len(resume_text), 400)]
    chunk_embeddings = embedding_model.encode(chunks)
    
    # Query: "What are the technical skills mentioned?"
    query = "technical skills"
    query_embedding = embedding_model.encode([query])[0]
    
    # Simple similarity search
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = "\n".join([chunks[i] for i in top_indices])
    
    prompt = f"""
    Based ONLY on the provided context, extract the technical skills.
    Return ONLY a comma-separated list of skills.
    
    Context:
    {context}
    """
    return get_ollama_response(prompt)

def run_graph_rag(resume_text, neo4j_mgr):
    # 1. First extract entities to build graph (Simplified for demo)
    # in a real system this would be more complex
    extraction_prompt = f"""
    Extract technical skills and their categories from this resume.
    Format as JSON: {{"skills": [{{"name": "Python", "category": "Programming"}}, ...]}}
    
    Resume:
    {resume_text}
    """
    extracted_json = get_ollama_response(extraction_prompt)
    try:
        # Basic cleanup of LLM response
        match = re.search(r'\{.*\}', extracted_json, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            
            # 2. Populate Neo4j
            neo4j_mgr.clear_db()
            for skill in data.get('skills', []):
                neo4j_mgr.run_query(
                    "MERGE (s:Skill {name: $name}) SET s.category = $cat",
                    {"name": skill['name'], "cat": skill['category']}
                )
            
            # 3. Query Graph for Facts
            facts = neo4j_mgr.run_query("MATCH (s:Skill) RETURN s.name as name, s.category as cat")
            context = ", ".join([f"{f['name']} ({f['cat']})" for f in facts])
            
            prompt = f"""
            You are a strict resume analyzer. Use ONLY the facts retrieved from our Knowledge Graph.
            If a skill is NOT in the facts, DO NOT mention it.
            
            Retrieved Facts:
            {context}
            
            List the skills identified. Return ONLY a comma-separated list.
            """
            return get_ollama_response(prompt)
    except:
        return "Error in GraphRAG pipeline"

# --- Evaluation ---

def evaluate_hallucination(generated_skills, resume_text):
    # Basic verification: check if skill name exists in original text
    skills = [s.strip() for s in generated_skills.split(',')]
    supported = []
    hallucinated = []
    
    for skill in skills:
        if not skill: continue
        # Case insensitive word match
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE):
            supported.append(skill)
        else:
            hallucinated.append(skill)
            
    faithfulness = len(supported) / len(skills) if skills else 0
    return faithfulness, supported, hallucinated

# --- Main UI ---

st.title("🚀 GraphRAG vs Hallucination")
st.markdown("### Experimental Evaluation of RAG Topologies")

with st.sidebar:
    st.header("Settings")
    ollama_status = st.empty()
    neo4j_status = st.empty()
    
    # Check Ollama
    try:
        ollama.list()
        ollama_status.success("Ollama Connected")
    except:
        ollama_status.error("Ollama NOT Connected")
        
    # Check Neo4j
    neo4j_mgr = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if neo4j_mgr.connected:
        neo4j_status.success("Neo4j Connected")
    else:
        neo4j_status.error(f"Neo4j Error: {neo4j_mgr.error}")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file and st.button("Run Experiment"):
    resume_text = extract_text_from_pdf(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Running No RAG..."):
        no_rag_results = run_no_rag(resume_text)
        f1, s1, h1 = evaluate_hallucination(no_rag_results, resume_text)
        
    with st.spinner("Running Vector RAG..."):
        vector_rag_results = run_vector_rag(resume_text, embedding_model)
        f2, s2, h2 = evaluate_hallucination(vector_rag_results, resume_text)
        
    with st.spinner("Running GraphRAG..."):
        graph_rag_results = run_graph_rag(resume_text, neo4j_mgr)
        f3, s3, h3 = evaluate_hallucination(graph_rag_results, resume_text)
        
    # Display Results
    with col1:
        st.subheader("❌ No RAG")
        st.write(no_rag_results)
        st.metric("Faithfulness", f"{f1:.2%}")
        if h1:
            st.error(f"Hallucinations: {', '.join(h1)}")
            
    with col2:
        st.subheader("⚠️ Vector RAG")
        st.write(vector_rag_results)
        st.metric("Faithfulness", f"{f2:.2%}")
        if h2:
            st.warning(f"Hallucinations: {', '.join(h2)}")
            
    with col3:
        st.subheader("✅ GraphRAG")
        st.write(graph_rag_results)
        st.metric("Faithfulness", f"{f3:.2%}")
        if h3:
            st.error(f"Hallucinations: {', '.join(h3)}")
        else:
            st.success("Zero Hallucination!")

    # Visualization
    st.divider()
    st.header("📊 Performance Comparison")
    
    fig = go.Figure(data=[
        go.Bar(name='Faithfulness', x=['No RAG', 'Vector RAG', 'GraphRAG'], y=[f1, f2, f3], marker_color=['#dc3545', '#ffc107', '#28a745'])
    ])
    fig.update_layout(title="Faithfulness Score Comparison", yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    # Graph Visualization (Simple NetworkX to Plotly)
    if f3 > 0:
        st.header("🕸️ GraphRAG Knowledge Map")
        facts = neo4j_mgr.run_query("MATCH (s:Skill) RETURN s.name as name")
        G = nx.Graph()
        for f in facts:
            G.add_edge("Resume", f['name'])
        
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), 
                                 textposition="top center", marker=dict(size=15, color='#007bff'))
        
        fig_graph = go.Figure(data=[edge_trace, node_trace])
        st.plotly_chart(fig_graph, use_container_width=True)

st.divider()
st.info("Developed for GraphRAG Research Demonstration")
