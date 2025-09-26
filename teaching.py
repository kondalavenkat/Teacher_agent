import streamlit as st
import ollama
import os
import requests
import json
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="🤖 AI Teaching Agent Team (Ollama)", layout="centered")

# Initialize session state for API keys and topic
if 'ollama_model' not in st.session_state:
    st.session_state['ollama_model'] = 'llama3.2:latest'
if 'composio_api_key' not in st.session_state:
    st.session_state['composio_api_key'] = ''
if 'serpapi_api_key' not in st.session_state:
    st.session_state['serpapi_api_key'] = ''
if 'topic' not in st.session_state:
    st.session_state['topic'] = ''

# Streamlit sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Ollama model selection
    st.session_state['ollama_model'] = st.selectbox(
       "Select Ollama Model",
        ['llama3.2:latest', 'mistral:latest', 'sqlcoder:latest', 'gemma2:2b', 'phi3:mini', 'tinyllama:latest'],
        help="Make sure the model is pulled in Ollama first"
    )
    
    st.session_state['composio_api_key'] = st.text_input("Enter your Composio API Key", type="password").strip()
    st.session_state['serpapi_api_key'] = st.text_input("Enter your SerpAPI Key", type="password").strip()

    # Add info about terminal responses
    st.info("Note: You can also view detailed agent responses\nin your terminal after execution.")

# Validate API keys
if not st.session_state['composio_api_key'] or not st.session_state['serpapi_api_key']:
    st.error("Please enter Composio and SerpAPI keys in the sidebar.")
    st.stop()

# Test Ollama connection
try:
    models = ollama.list()
    available_models = [model['model'] for model in models['models']]
    if st.session_state['ollama_model'] not in available_models:
        st.error(f"Model {st.session_state['ollama_model']} not found. Available models: {', '.join(available_models)}")
        st.stop()
except Exception as e:
    st.error(f"Error connecting to Ollama: {e}")
    st.stop()

# Helper function to call Ollama
def call_ollama(prompt: str, model: str = None) -> str:
    """Call Ollama with the given prompt"""
    try:
        response = ollama.chat(
            model=model or st.session_state['ollama_model'],
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error calling Ollama: {e}"

# Helper function for SerpAPI search
def search_web(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Search the web using SerpAPI"""
    try:
        params = {
            'q': query,
            'api_key': api_key,
            'engine': 'google',
            'num': 5
        }
        response = requests.get('https://serpapi.com/search', params=params)
        data = response.json()
        
        results = []
        if 'organic_results' in data:
            for result in data['organic_results']:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                })
        return results
    except Exception as e:
        return [{'error': f"Search error: {e}"}]

# Professor Agent
def professor_agent(topic: str) -> Dict[str, str]:
    """Research and Knowledge Specialist"""
    prompt = f"""
    You are a Professor and Research Specialist. Create a comprehensive knowledge base for the topic: {topic}
    
    Requirements:
    1. Explain the topic from first principles
    2. Include key terminology, core principles, and practical applications
    3. Make it detailed and accessible for beginners
    4. Format it clearly with headings and structure
    5. Include current developments and trends
    
    Create a detailed report that anyone starting out can read and get maximum value from.
    """
    
    response = call_ollama(prompt)
    
    return {
        'content': response,
        'title': f"Professor Report - {topic}",
        'filename': f"professor_report_{topic.lower().replace(' ', '_')}"
    }

# Academic Advisor Agent
def academic_advisor_agent(topic: str) -> Dict[str, str]:
    """Learning Path Designer"""
    prompt = f"""
    You are an Academic Advisor and Learning Path Designer. Create a detailed learning roadmap for: {topic}
    
    Requirements:
    1. Break down the topic into logical subtopics
    2. Arrange them in order of progression
    3. Include estimated time commitments for each section
    4. Create a structured learning path to become an expert
    5. Include milestones and checkpoints
    
    Present the roadmap in a clear, structured format with timelines.
    """
    
    response = call_ollama(prompt)
    
    return {
        'content': response,
        'title': f"Learning Roadmap - {topic}",
        'filename': f"learning_roadmap_{topic.lower().replace(' ', '_')}"
    }

# Research Librarian Agent
def research_librarian_agent(topic: str) -> Dict[str, Any]:
    """Learning Resource Specialist"""
    # Search for resources
    search_results = search_web(f"learn {topic} tutorial course", st.session_state['serpapi_api_key'])
    
    prompt = f"""
    You are a Research Librarian. Based on the search results below, curate high-quality learning resources for: {topic}
    
    Search Results:
    {json.dumps(search_results, indent=2)}
    
    Requirements:
    1. Evaluate and recommend the best resources
    2. Categorize them by type (tutorials, courses, books, videos, etc.)
    3. Include difficulty levels and prerequisites
    4. Provide brief descriptions and why each resource is valuable
    5. Include both free and paid options
    
    Create a comprehensive resource guide.
    """
    
    response = call_ollama(prompt)
    
    return {
        'content': response,
        'title': f"Learning Resources - {topic}",
        'filename': f"learning_resources_{topic.lower().replace(' ', '_')}",
        'search_results': search_results
    }

# Teaching Assistant Agent
def teaching_assistant_agent(topic: str) -> Dict[str, str]:
    """Practice Materials Creator"""
    prompt = f"""
    You are a Teaching Assistant. Create practice materials, exercises, and projects for: {topic}
    
    Requirements:
    1. Create hands-on exercises and practice problems
    2. Design projects that build real-world skills
    3. Include coding challenges if applicable
    4. Provide step-by-step instructions
    5. Include assessment criteria and solutions
    6. Make materials progressive from beginner to advanced
    
    Create comprehensive practice materials that reinforce learning.
    """
    
    response = call_ollama(prompt)
    
    return {
        'content': response,
        'title': f"Practice Materials - {topic}",
        'filename': f"practice_materials_{topic.lower().replace(' ', '_')}"
    }

# Main Streamlit UI
st.title("🤖 AI Teaching Agent Team (Ollama)")
st.markdown("Get personalized learning plans powered by Ollama models!")

# Topic input
topic = st.text_input("Enter a topic you want to learn about:", placeholder="e.g., Python Programming, Machine Learning, Data Science")

if st.button("Generate Learning Plan", type="primary"):
    if not topic:
        st.error("Please enter a topic!")
    else:
        st.session_state['topic'] = topic
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Professor is researching..."):
                professor_result = professor_agent(topic)
            
            with st.spinner("Academic Advisor is planning..."):
                advisor_result = academic_advisor_agent(topic)
        
        with col2:
            with st.spinner("Research Librarian is curating resources..."):
                librarian_result = research_librarian_agent(topic)
            
            with st.spinner("Teaching Assistant is creating practice materials..."):
                assistant_result = teaching_assistant_agent(topic)
        
        # Create download section
        st.markdown("### 📄 Download Learning Materials:")
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Download buttons in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📚 Professor Report",
                data=professor_result['content'],
                file_name=f"{professor_result['filename']}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="🗺️ Learning Roadmap",
                data=advisor_result['content'],
                file_name=f"{advisor_result['filename']}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                label="📋 Resources Guide",
                data=librarian_result['content'],
                file_name=f"{librarian_result['filename']}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col4:
            st.download_button(
                label="🎯 Practice Materials",
                data=assistant_result['content'],
                file_name=f"{assistant_result['filename']}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Complete learning package download
        st.markdown("---")
        complete_package = f"""
COMPLETE LEARNING PACKAGE FOR: {topic.upper()}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==========================================
📚 PROFESSOR'S KNOWLEDGE BASE
==========================================
{professor_result['content']}

==========================================
🗺️ ACADEMIC ADVISOR'S LEARNING ROADMAP
==========================================
{advisor_result['content']}

==========================================
📋 RESEARCH LIBRARIAN'S RESOURCE GUIDE
==========================================
{librarian_result['content']}

==========================================
🎯 TEACHING ASSISTANT'S PRACTICE MATERIALS
==========================================
{assistant_result['content']}

==========================================
END OF LEARNING PACKAGE
==========================================
        """
        
        st.download_button(
            label="📦 Download Complete Learning Package",
            data=complete_package,
            file_name=f"complete_learning_package_{topic.lower().replace(' ', '_')}_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Display results
        st.markdown("### 🎓 Professor's Knowledge Base:")
        st.markdown(professor_result['content'])
        st.divider()
        
        st.markdown("### 🗺️ Academic Advisor's Learning Roadmap:")
        st.markdown(advisor_result['content'])
        st.divider()
        
        st.markdown("### 📚 Research Librarian's Resource Guide:")
        st.markdown(librarian_result['content'])
        st.divider()
        
        st.markdown("### 🎯 Teaching Assistant's Practice Materials:")
        st.markdown(assistant_result['content'])
        st.divider()

# Information about the agents
st.markdown("---")
st.markdown("### About the Agents:")
st.markdown("""
- **Professor**: Researches the topic and creates a detailed knowledge base using Ollama
- **Academic Advisor**: Designs a structured learning roadmap for the topic
- **Research Librarian**: Curates high-quality learning resources using web search
- **Teaching Assistant**: Creates practice materials, exercises, and projects
""")

# Model information
st.markdown("### 🤖 Current Model:")
st.info(f"Using Ollama model: **{st.session_state['ollama_model']}**")

# Instructions
st.markdown("### 📋 Instructions:")
st.markdown("""
1. **Select an Ollama model** from the sidebar (make sure it's pulled first)
2. **Enter your API keys** for Composio and SerpAPI
3. **Type a topic** you want to learn about
4. **Click Generate Learning Plan** and wait for the agents to work
5. **Download the learning materials** using the download buttons
6. **View the results** displayed below
""")