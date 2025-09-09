"""
Improved Startup Trend Analysis Pipeline with RAG
Enhanced error handling, performance optimizations, and simplified features
"""

import os
import glob
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

# External libraries
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from openai import OpenAI
import anthropic
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import streamlit as st
from collections import Counter
import numpy as np

# ============ Enhanced Configuration ============
@dataclass
class Config:
    """Centralized configuration with validation"""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    use_anthropic: bool = False
    openai_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"
    embed_model: str = "text-embedding-3-small"
    persist_dir: str = "./chroma_startup_trends"
    collection_name: str = "startup_trends_2025"
    corpus_dir: str = "./corpus"

    max_chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200
    max_results: int = 6
    max_tokens: int = 1000
    temperature: float = 0.1
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment"""
        load_dotenv()
        config = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        config.use_anthropic = bool(config.anthropic_api_key)
        config.validate()
        return config
    
    def validate(self):
        """Validate configuration"""
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("At least one API key (OpenAI or Anthropic) must be provided")
        
        # Create directories if they don't exist
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.corpus_dir).mkdir(parents=True, exist_ok=True)
        
        if not Path(self.corpus_dir).exists():
            logging.warning(f"Corpus directory {self.corpus_dir} does not exist")

# ============ Enhanced Logging ============
def setup_logging(level: str = "INFO"):
    """Configure logging with better formatting"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============ Document Processing with Chunking ============
class DocumentProcessor:
    """Enhanced document processing with chunking and caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @lru_cache(maxsize=128)
    def read_file(self, path: str) -> Optional[str]:
        """Read file with caching and better error handling"""
        ext = Path(path).suffix.lower()
        
        try:
            if ext in ('.txt', '.md', '.csv', '.json'):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif ext == '.pdf':
                return self._read_pdf(path)
            else:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to read {path}: {e}")
            return None
    
    def _read_pdf(self, path: str) -> str:
        """Extract text from PDF with error handling"""
        try:
            reader = PdfReader(path)
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {i} from {path}: {e}")
            return "\n".join(pages)
        except Exception as e:
            self.logger.error(f"Failed to read PDF {path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better context"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few sentences for overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        overlap_size += len(s)
                        overlap_chunk.insert(0, s)
                        if overlap_size >= self.config.chunk_overlap:
                            break
                    current_chunk = overlap_chunk
                    current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def load_corpus(self) -> List[Dict[str, str]]:
        """Load and process corpus with parallel processing"""
        docs = []
        files = list(Path(self.config.corpus_dir).rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            self.logger.warning(f"No files found in {self.config.corpus_dir}")
            return docs
        
        self.logger.info(f"Processing {len(files)} files...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self._process_file, str(f)): f 
                for f in files
            }
            
            for future in as_completed(future_to_file):
                file_docs = future.result()
                if file_docs:
                    docs.extend(file_docs)
        
        self.logger.info(f"Loaded {len(docs)} document chunks")
        return docs
    
    def _process_file(self, path: str) -> List[Dict[str, str]]:
        """Process a single file into chunks"""
        content = self.read_file(path)
        if not content or not content.strip():
            return []
        
        chunks = self.chunk_text(content)
        return [
            {
                "id": f"{path}_chunk_{i}",
                "text": chunk,
                "source": path,
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]

# ============ Enhanced Vector Store ============
class VectorStore:
    """Improved vector store with better error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self.embed_fn = None
        
    def initialize(self) -> 'VectorStore':
        """Initialize vector store with error handling"""
        try:
            self.client = chromadb.PersistentClient(path=self.config.persist_dir)
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.openai_api_key,
                model_name=self.config.embed_model
            )
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection(
                    self.config.collection_name,
                    embedding_function=self.embed_fn
                )
                self.logger.info(f"Loaded collection with {self.collection.count()} documents")
            except Exception:
                self.collection = self.client.create_collection(
                    self.config.collection_name,
                    embedding_function=self.embed_fn
                )
                self.logger.info("Created new collection")
                
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, str]]) -> bool:
        """Index documents with batch processing"""
        if not documents:
            self.logger.warning("No documents to index")
            return False
        
        try:
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                self.collection.add(
                    ids=[d["id"] for d in batch],
                    documents=[d["text"] for d in batch],
                    metadatas=[
                        {
                            "source": d["source"],
                            "chunk_index": d.get("chunk_index", 0)
                        }
                        for d in batch
                    ]
                )
                
                self.logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index documents: {e}")
            return False
    
    def search(self, query: str, n_results: int = 6) -> List[Dict[str, Any]]:
        """Search with error handling and result formatting"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            hits = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    hits.append({
                        "text": doc[:1500],  # Limit text length
                        "source": results["metadatas"][0][i].get("source", "unknown"),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                        "distance": float(results["distances"][0][i])
                    })
            
            return hits
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

# ============ Enhanced LLM Wrapper ============
class LLMClient:
    """Unified LLM client with retry logic and error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.openai_client = None
        self.anthropic_client = None
        
        if config.openai_api_key:
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        if config.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    
    def call(self, system: str, user: str, max_tokens: int = None, retries: int = 3) -> str:
        """Call LLM with retry logic"""
        max_tokens = max_tokens or self.config.max_tokens
        
        for attempt in range(retries):
            try:
                if self.config.use_anthropic and self.anthropic_client:
                    return self._call_anthropic(system, user, max_tokens)
                elif self.openai_client:
                    return self._call_openai(system, user, max_tokens)
                else:
                    raise ValueError("No LLM client available")
                    
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    self.logger.error(f"All LLM call attempts failed")
                    return "Error: Failed to get LLM response"
        
        return "Error: Maximum retries exceeded"
    
    def _call_openai(self, system: str, user: str, max_tokens: int) -> str:
        """Call OpenAI API"""
        response = self.openai_client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=self.config.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        """Call Anthropic API"""
        response = self.anthropic_client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

# ============ Enhanced Graph State ============
class GraphState(TypedDict):
    question: str
    rag_hits: List[Dict[str, Any]]
    research_summary: str
    analysis_brief: str
    keywords: List[str]
    error_messages: List[str]

# ============ Enhanced Agents ============
class ResearchAgent:
    """Enhanced research agent with better keyword extraction"""
    
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def process(self, state: GraphState) -> GraphState:
        """Process research query with error handling"""
        try:
            question = state["question"]
            
            # Search for relevant documents
            hits = self.vector_store.search(question, n_results=6)
            
            if not hits:
                state["error_messages"].append("No relevant documents found")
                state["research_summary"] = "No relevant information found in the knowledge base."
                return state
            
            # Create context from hits
            context = self._format_context(hits)
            
            # Generate summary
            system_prompt = """You are a neutral startup research analyst.
            Summarize key 2025 trends & signals strictly from the provided snippets.
            Use short bullets and include a final 'Sources:' list of filenames.
            If evidence is weak or conflicting, say so."""
            
            user_prompt = f"Question:\n{question}\n\nSnippets:\n{context}"
            
            summary = self.llm_client.call(system_prompt, user_prompt, max_tokens=1200)
            
            # Extract keywords more intelligently
            keywords = self._extract_keywords(summary)
            
            state["rag_hits"] = hits
            state["research_summary"] = summary
            state["keywords"] = keywords
            
        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            state["error_messages"].append(f"Research failed: {str(e)}")
            state["research_summary"] = "Research process encountered an error."
        
        return state
    
    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        """Format search hits into context"""
        context_parts = []
        for hit in hits:
            source_name = Path(hit['source']).name
            text = hit['text']
            context_parts.append(f"[{source_name}] {text}")
        return "\n\n".join(context_parts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords using TF-IDF-like approach"""
        # Simple keyword extraction (can be enhanced with NLTK or spaCy)
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-\+]{2,}\b', text.lower())
        
        # Enhanced stop words
        stop_words = set("""
            the a an and or but to for from with without into of on at by as is are was were
            be being been if in this that these those there here then thus hence about above
            below more most less least not no yes data ai ml gpt llm apis api model models
            trend trends can will would should could may might must shall
        """.split())
        
        # Filter and count
        word_counts = Counter(w for w in words if w not in stop_words and len(w) > 3)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(50)]

class AnalystAgent:
    """Enhanced analyst agent with structured output"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def process(self, state: GraphState) -> GraphState:
        """Generate analysis with error handling"""
        try:
            if not state.get("research_summary"):
                state["analysis_brief"] = "No research summary available for analysis."
                return state
            
            system_prompt = """You are a pragmatic business analyst for startups in 2025.
            Create an action-oriented brief grounded in the research summary.
            Use numbered lists. Keep sentences tight. Include measurable KPIs."""
            
            user_prompt = f"""Question: {state['question']}

Research summary:
{state['research_summary']}

Deliverables:
1) Top opportunities (why now)
2) Key risks + mitigations
3) 90-day KPI set (leading & lagging)
4) 3–5 testable hypotheses (A/B-style)
5) 6-sentence executive brief"""
            
            brief = self.llm_client.call(system_prompt, user_prompt, max_tokens=1300)
            state["analysis_brief"] = brief
            
        except Exception as e:
            self.logger.error(f"Analyst agent failed: {e}")
            state["error_messages"].append(f"Analysis failed: {str(e)}")
            state["analysis_brief"] = "Analysis process encountered an error."
        
        return state

# ============ Streamlit Interface ============
def display_results_streamlit(state: GraphState):
    """Display results using Streamlit interface with enhanced visualizations"""
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "🔍 Research Details", "📈 Trend Analysis", "📚 RAG Sources"])
    
    with tab1:
        st.header("Executive Analysis")
        analysis_brief = state.get('analysis_brief', 'No analysis available')
        if analysis_brief and analysis_brief != 'No analysis available':
            st.markdown(analysis_brief)
        else:
            st.warning("No analysis brief available")
    
    with tab2:
        st.header("Research Summary")
        research_summary = state.get('research_summary', 'No research summary available')
        if research_summary and research_summary != 'No research summary available':
            st.markdown(research_summary)
        else:
            st.warning("No research summary available")
        
        # Show error messages if any
        if state.get('error_messages'):
            st.subheader("⚠️ Warnings")
            for msg in state['error_messages']:
                st.warning(msg)
    
    with tab3:
        st.header("📈 Trend Analysis & Keywords")
        
        # Keywords analysis
        keywords = state.get('keywords', [])
        if keywords:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Keywords by Frequency")
                keyword_counts = Counter(keywords).most_common(15)
                if keyword_counts:
                    import pandas as pd
                    
                    # Create DataFrame for better visualization
                    df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Frequency'])
                    
                    # Create horizontal bar chart
                    st.bar_chart(df.set_index('Keyword')['Frequency'], horizontal=True)
                    
                    # Show top keywords as metrics
                    st.subheader("Key Trend Indicators")
                    top_keywords = keyword_counts[:6]
                    cols = st.columns(3)
                    for i, (keyword, freq) in enumerate(top_keywords):
                        with cols[i % 3]:
                            st.metric(label=keyword.title(), value=f"{freq}x")
            
            with col2:
                st.subheader("Keyword Cloud")
                # Create a simple text-based keyword display
                all_keywords = [kw for kw, _ in keyword_counts]
                keyword_text = " • ".join(all_keywords[:20])
                st.info(f"**Trending Terms:**\n\n{keyword_text}")
                
                # Trend strength indicator
                total_mentions = sum(freq for _, freq in keyword_counts)
                st.metric("Total Keyword Mentions", total_mentions)
        else:
            st.info("No keywords extracted from the analysis")
    
    with tab4:
        st.header("📚 RAG Sources & Relevance")
        rag_hits = state.get('rag_hits', [])
        if rag_hits:
            # Create relevance visualization
            st.subheader("Source Relevance Analysis")
            
            import pandas as pd
            
            # Prepare data for visualization
            source_data = []
            for i, hit in enumerate(rag_hits, 1):
                source_name = Path(hit['source']).name
                relevance = (1 - hit['distance']) * 100  # Convert to percentage
                source_data.append({
                    'Source': source_name,
                    'Relevance %': relevance,
                    'Rank': i
                })
            
            df_sources = pd.DataFrame(source_data)
            
            # Display relevance chart
            st.bar_chart(df_sources.set_index('Source')['Relevance %'])
            
            # Display sources with expandable content
            st.subheader("Source Details")
            for i, hit in enumerate(rag_hits, 1):
                source_name = Path(hit['source']).name
                relevance = (1 - hit['distance']) * 100
                
                # Color code by relevance
                if relevance >= 80:
                    relevance_color = "🟢"
                elif relevance >= 60:
                    relevance_color = "🟡"
                else:
                    relevance_color = "🔴"
                
                with st.expander(f"{relevance_color} **#{i}** {source_name} - **{relevance:.1f}% relevant**"):
                    # Show chunk info
                    st.caption(f"Chunk {hit.get('chunk_index', 0)} • Distance: {hit['distance']:.3f}")
                    
                    # Show content with better formatting
                    content = hit['text']
                    if len(content) > 800:
                        st.markdown(f"**Preview:**\n\n{content[:800]}...")
                        
                        with st.expander("Show full content"):
                            st.text(content)
                    else:
                        st.markdown(f"**Content:**\n\n{content}")
            
            # Summary metrics
            st.subheader("RAG Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_relevance = sum((1 - hit['distance']) * 100 for hit in rag_hits) / len(rag_hits)
                st.metric("Average Relevance", f"{avg_relevance:.1f}%")
            
            with col2:
                unique_sources = len(set(Path(hit['source']).name for hit in rag_hits))
                st.metric("Unique Sources", unique_sources)
            
            with col3:
                high_relevance = sum(1 for hit in rag_hits if (1 - hit['distance']) > 0.7)
                st.metric("High Relevance Sources", f"{high_relevance}/{len(rag_hits)}")
                
        else:
            st.info("No RAG sources available")

# ============ Main Pipeline ============
class StartupTrendPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.vector_store = None
        self.llm_client = None
        self.graph = None
    
    def initialize(self) -> 'StartupTrendPipeline':
        """Initialize all components"""
        self.logger.info("Initializing pipeline...")
        
        # Initialize vector store
        self.vector_store = VectorStore(self.config).initialize()
        
        # Check if we need to index documents
        if self.vector_store.collection.count() == 0:
            self.logger.info("Empty collection, indexing documents...")
            processor = DocumentProcessor(self.config)
            documents = processor.load_corpus()
            
            if documents:
                self.vector_store.index_documents(documents)
            else:
                self.logger.warning("No documents found to index")
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config)
        
        # Build graph
        self.graph = self._build_graph()
        
        self.logger.info("Pipeline initialized successfully")
        return self
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        graph = StateGraph(GraphState)
        
        # Create agents
        research_agent = ResearchAgent(self.vector_store, self.llm_client)
        analyst_agent = AnalystAgent(self.llm_client)
        
        # Add nodes
        graph.add_node("research", research_agent.process)
        graph.add_node("analysis", analyst_agent.process)
        
        # Add edges
        graph.add_edge(START, "research")
        graph.add_edge("research", "analysis")
        graph.add_edge("analysis", END)
        
        return graph.compile()
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the complete pipeline"""
        self.logger.info(f"Running pipeline for question: {question[:100]}...")
        
        # Initialize state
        initial_state: GraphState = {
            "question": question,
            "rag_hits": [],
            "research_summary": "",
            "analysis_brief": "",
            "keywords": [],
            "error_messages": []
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Log results
            self.logger.info("Pipeline completed successfully")
            
            return {
                "success": True,
                "state": final_state
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "state": initial_state
            }
    


# ============ Streamlit App ============
def main():
    """Streamlit app main function"""
    st.set_page_config(
        page_title="Startup Trend Analysis",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Header
    st.title("🚀 Startup Trend Analysis Pipeline")
    st.markdown("Analyze startup trends using AI-powered research and RAG")
    
    # Auto-initialize pipeline on first load
    if st.session_state.pipeline is None:
        try:
            with st.spinner("Initializing pipeline..."):
                # Load configuration from environment
                config = Config.from_env()
                st.session_state.pipeline = StartupTrendPipeline(config).initialize()
            st.success("Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.info("Please ensure your API keys are set in environment variables (.env file)")
            st.stop()
    
    # Sidebar with pipeline info
    with st.sidebar:
        st.header("Pipeline Status")
        if st.session_state.pipeline:
            st.success("✅ Pipeline Ready")
            config = st.session_state.pipeline.config
            st.info(f"🤖 Model: {'Anthropic Claude' if config.use_anthropic else 'OpenAI GPT'}")
            st.info(f"📚 Documents: {st.session_state.pipeline.vector_store.collection.count()}")
        
        if st.button("🔄 Reinitialize Pipeline"):
            st.session_state.pipeline = None
            st.rerun()
    
    # Main interface
    if st.session_state.pipeline:
        st.header("Ask Your Question")
        
        # Question input
        default_question = "What are the most investable AI-native B2B startup trends for 2025, considering regulation, distribution, and unit economics?"
        question = st.text_area(
            "Enter your startup trend question:",
            value=default_question,
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Analyze", type="primary"):
                if question.strip():
                    try:
                        with st.spinner("Running analysis..."):
                            result = st.session_state.pipeline.run(question)
                        
                        if result["success"]:
                            st.session_state.results = result["state"]
                            st.success("Analysis completed successfully!")
                        else:
                            st.error(f"Analysis failed: {result.get('error')}")
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.results = None
                st.rerun()
        
        # Display results
        if st.session_state.results:
            st.markdown("---")
            display_results_streamlit(st.session_state.results)

if __name__ == "__main__":
    main()