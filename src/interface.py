import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from services.recommendation_service import recommend_books, test_vector_store
from core.text_preprocessor import preprocessing_pipeline
from pipeline import run_data_pipeline

books_file = "../data/books.csv"

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load existing data if available"""
    try:
        if os.path.exists("../data/books.csv"):
            return pd.read_csv("../data/books.csv")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def load_embeddings_data():
    """Load embeddings data if available"""
    try:
        if os.path.exists("../data/books_embeddings.csv"):
            df = pd.read_csv("../data/books_embeddings.csv")
            
            # Handle embeddings that might be stored as strings
            if 'embedding_str' in df.columns:
                # Convert string embeddings back to lists
                df['embedding'] = df['embedding_str'].apply(lambda x: [float(i) for i in x.split(',')])
            elif 'embedding' in df.columns:
                # Check if embeddings are already lists
                sample_embedding = df['embedding'].iloc[0]
                if isinstance(sample_embedding, str):
                    # Convert string embeddings back to lists
                    df['embedding'] = df['embedding'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
            
            return df
        return None
    except Exception as e:
        st.error(f"Error loading embeddings data: {str(e)}")
        return None

def create_dashboard_metrics(df):
    """Create dashboard metrics"""
    if df is None or df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", len(df))
    
    with col2:
        unique_authors = df['authors'].nunique()
        st.metric("Unique Authors", unique_authors)
    
    with col3:
        unique_categories = df['primary_category'].nunique()
        st.metric("Categories", unique_categories)
    
    with col4:
        avg_description_length = df['description'].str.len().mean()
        st.metric("Avg Description Length", f"{avg_description_length:.0f} chars")

def create_visualizations(df):
    """Create various visualizations"""
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["📊 Categories", "👥 Authors"])
    
    with tab1:
        st.subheader("Book Categories Distribution")
        category_counts = df['primary_category'].value_counts().head(10)
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Book Categories",
            labels={'x': 'Number of Books', 'y': 'Category'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Top Authors")
        # Split authors and count
        all_authors = []
        for authors_str in df['authors'].dropna():
            authors = [author.strip() for author in authors_str.split(',')]
            all_authors.extend(authors)
        
        author_counts = pd.Series(all_authors).value_counts().head(10)
        fig = px.bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            title="Top 10 Authors",
            labels={'x': 'Number of Books', 'y': 'Author'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def recommendation_interface():
    """Interface for the recommendation service"""
    st.header("🔍 Book Recommendation Service")
    
    # Check if vector store exists
    if not os.path.exists("../book_index"):
        st.error("❌ Vector store not found. Please run the automated pipeline first.")
        st.info("💡 To create the vector store:")
        st.write("1. Go to '🔗 Automated Chains'")
        st.write("2. Click '🔄 Run LangChain Pipeline'")
        st.write("3. This will automatically create the vector store")
        return
    
    # Check if vector store files exist
    if not os.path.exists("../book_index/index.faiss") or not os.path.exists("../book_index/index.pkl"):
        st.error("❌ Vector store files incomplete. Please recreate the vector store.")
        return
    
    # Query input section
    st.subheader("Search for Books")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'machine learning', 'science fiction', 'history of ancient civilizations'",
            help="Describe what kind of books you're looking for"
        )
    
    with col2:
        k = st.number_input(
            "Number of recommendations:",
            min_value=1,
            max_value=20,
            value=5,
            help="How many books to recommend"
        )
    
    # Advanced search options
    st.subheader("🔧 Advanced Search Options")
    
    # Vertical layout for search options
    use_preprocessed = st.checkbox("Use Preprocessed Search", value=True,
                                 help="Apply same preprocessing to query as used for embeddings")
    
    # Test vector store button
    if st.button("🧪 Test Vector Store"):
        with st.spinner("Testing vector store..."):
            try:
                if test_vector_store():
                    st.success("✅ Vector store is working!")
                else:
                    st.error("❌ Vector store test failed")
            except Exception as e:
                st.error(f"❌ Test error: {str(e)}")
    
    # Search button
    if st.button("🔍 Search", type="primary", disabled=not query):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        else:
            with st.spinner("Searching for recommendations..."):
                try:
                    # Preprocess query if requested
                    search_query = query
                    if use_preprocessed:
                        try:
                            preprocessor = preprocessing_pipeline()
                            processed_query = preprocessor.preprocess_text(query)
                            st.info(f"🔤 Query preprocessed: '{query}' → '{processed_query}'")
                            search_query = processed_query
                        except Exception as e:
                            st.warning(f"Could not preprocess query: {str(e)}")
                            search_query = query
                    
                    # Add debugging information
                    st.info(f"🔍 Searching for: '{search_query}'")
                    st.info(f"📊 Requesting {k} recommendations")
                    
                    
                    
                    try:
                        results = recommend_books(search_query, k)
                        
                        if results.empty:
                            st.warning("No results found. Try a different query or check if the vector store exists.")
                            return
                    except Exception as e:
                        st.error(f"❌ Error in recommendation service: {str(e)}")
                        st.info("💡 Troubleshooting tips:")
                        st.write("- Check if vector store exists in ../book_index/")
                        st.write("- Verify OpenAI API key is set")
                        st.write("- Ensure embeddings were generated successfully")
                        return
                    
                    # Use all results (no filtering)
                    results_filtered = results
                    
                    if not results_filtered.empty:
                        
                        # Display results summary
                        st.subheader("📚 Recommended Books")
                        
                        # Results summary
                        st.info(f"🎯 Found {len(results_filtered)} recommendations for your query")
                        
                        # Show category distribution
                        if 'primary_category' in results_filtered.columns:
                            category_counts = results_filtered['primary_category'].value_counts()
                            if len(category_counts) > 1:
                                st.write("**📊 Category Distribution:**")
                                category_text = ", ".join([f"{cat} ({count})" for cat, count in category_counts.head(3).items()])
                                st.write(category_text)
                        
                        st.markdown("---")
                        
                        for idx, row in results_filtered.iterrows():
                            # Create color-coded similarity score
                            score = row['similarity_score']
                            if score >= 0.8:
                                score_color = "🟢"
                                score_text = f"**{score:.3f}** (Excellent Match)"
                            elif score >= 0.6:
                                score_color = "🟡"
                                score_text = f"**{score:.3f}** (Good Match)"
                            elif score >= 0.4:
                                score_color = "🟠"
                                score_text = f"**{score:.3f}** (Fair Match)"
                            else:
                                score_color = "🔴"
                                score_text = f"**{score:.3f}** (Weak Match)"
                            
                            with st.expander(f"{score_color} {row['title']} by {row['authors']}"):
                                # Similarity score and basic info
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.write("**Similarity Score:**")
                                    st.write(score_text)
                                    
                                    st.write("**Category:**")
                                    if pd.notna(row['categories']) and row['categories']:
                                        try:
                                            categories = str(row['categories']).split(', ')
                                            category_text = ", ".join([f"• {cat}" for cat in categories[:3]])
                                            if len(categories) > 3:
                                                category_text += f", ... and {len(categories) - 3} more"
                                            st.write(category_text)
                                        except (AttributeError, TypeError):
                                            st.write("Categories format error")
                                    else:
                                        st.write("No categories available")
                                
                                with col2:
                                    # Always show the current description (preprocessed or original)
                                    if use_preprocessed:
                                        st.write("**Preprocessed Description (used for embeddings):**")
                                        st.info("ℹ️ This is the preprocessed description used for embeddings")
                                    else:
                                        st.write("**Original Description:**")
                                        st.info("ℹ️ This is the original description")
                                    
                                    # Show full description without truncation
                                    description = row['description']
                                    st.write(description)
                                    
                                    # If using preprocessed, also show original description
                                    if use_preprocessed:
                                        st.markdown("---")
                                        st.write("**Original Description:**")
                                        try:
                                            original_df = load_data()
                                            if original_df is not None:
                                                original_book = original_df[original_df['id'] == row['id']]
                                                if not original_book.empty:
                                                    original_desc = original_book.iloc[0]['description']
                                                    st.write(original_desc)
                                                else:
                                                    st.warning("⚠️ Original description not found")
                                        except Exception:
                                            st.warning("⚠️ Original description not available")
                                
                                                                # Metadata section
                                st.markdown("---")
                                st.subheader("📋 Book Metadata")
                                
                                col3, col4 = st.columns(2)
                                
                                with col3:
                                    st.write("**📖 Book Information:**")
                                    st.write(f"**ID:** {row['id']}")
                                    st.write(f"**Title:** {row['title']}")
                                    st.write(f"**Authors:** {row['authors']}")
                                
                                with col4:
                                    st.write("**🏷️ Categories:**")
                                    if pd.notna(row['categories']) and row['categories']:
                                        try:
                                            categories = str(row['categories']).split(', ')
                                            for cat in categories[:3]:  # Show first 3 categories
                                                st.write(f"• {cat}")
                                            if len(categories) > 3:
                                                st.write(f"• ... and {len(categories) - 3} more")
                                        except (AttributeError, TypeError):
                                            st.write("Categories format error")
                                    else:
                                        st.write("No categories available")
                        
                        # Export results option
                        st.markdown("---")
                        st.subheader("💾 Export Results")
                        
                        # Create export dataframe with clean metadata (no collection info)
                        export_df = results_filtered.copy()
                        if 'similarity_score' in export_df.columns:
                            export_df['similarity_score'] = export_df['similarity_score'].round(4)
                        
                        # Remove collection info columns from export
                        columns_to_remove = ['search_strategy', 'retrieval_timestamp']
                        export_df = export_df.drop(columns=[col for col in columns_to_remove if col in export_df.columns])
                        
                        # Download button
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"book_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download all recommendations with metadata"
                        )
                        
                        # Show sample of export data
                        with st.expander("📋 Preview Export Data"):
                            st.dataframe(export_df, use_container_width=True)
                        
                    else:
                        st.warning("No recommendations found.")
                        
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")

def documentation_page():
    """Documentation page describing system architecture and techniques"""
    st.header("📚 System Documentation")
    
    # Table of contents
    st.sidebar.markdown("## 📋 Table of Contents")
    toc = st.sidebar.radio(
        "Jump to section:",
        ["🏗️ Architecture Overview", "🔗 LangChain Framework", "📊 Data Pipeline", "🔍 Recommendation Engine", "🔤 Text Preprocessing", "🗄️ Vector Store", "⚙️ Configuration", "🚀 API Integration"]
    )
    
    if toc == "🏗️ Architecture Overview":
        st.subheader("🏗️ System Architecture")
        
        st.write("""
        The Book Recommendation System is built with a modular architecture that separates data collection, 
        processing, and recommendation generation into distinct components.
        """)
        
        # Architecture diagram
        st.write("### System Components")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**📥 Data Collection**\n\nGoogle Books API\nSystematic data gathering\nRate limiting & error handling")
        
        with col2:
            st.info("**🔧 Processing Pipeline**\n\nText preprocessing\nEmbedding generation\nVector store creation")
        
        with col3:
            st.info("**🔍 Recommendation Engine**\n\nSemantic search\nSimilarity scoring\nResult ranking")
        
        st.write("### Data Flow")
        st.code("""
Google Books API → Raw Data → Text Preprocessing → Embeddings → FAISS Vector Store → Recommendations
        """, language="text")
        
        st.write("### Key Technologies")
        st.write("- **Google Books API**: Data collection")
        st.write("- **OpenAI Embeddings**: Semantic vector generation")
        st.write("- **FAISS**: High-performance similarity search")
        st.write("- **NLTK**: Text preprocessing and NLP")
        st.write("- **LangChain**: Modular processing chains")
        st.write("- **Streamlit**: User interface")
        st.write("- **Pandas**: Data manipulation")
        
    elif toc == "🔗 LangChain Framework":
        st.subheader("🔗 LangChain Framework Integration")
        
        st.write("""
        The system leverages LangChain's powerful chain architecture to create a modular, 
        monitored, and extensible data processing pipeline. LangChain provides the foundation 
        for building complex workflows with built-in monitoring and error handling.
        """)
        
        # LangChain Overview
        st.write("### What is LangChain?")
        st.write("""
        LangChain is a framework for developing applications powered by language models. 
        It provides a set of tools and abstractions for building complex AI applications, 
        including chains, agents, and memory systems.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**LangChain Benefits:**")
            st.write("- **Modular Design**: Reusable components")
            st.write("- **Built-in Monitoring**: Execution tracking")
            st.write("- **Error Handling**: Graceful failure recovery")
            st.write("- **Extensibility**: Easy to add new chains")
            st.write("- **Composability**: Chain combination and reordering")
        
        with col2:
            st.write("**Chain Features:**")
            st.write("- **Input/Output Validation**: Type checking")
            st.write("- **Execution Statistics**: Performance metrics")
            st.write("- **Batch Processing**: Efficient data handling")
            st.write("- **Configuration**: Flexible parameters")
            st.write("- **Logging**: Comprehensive execution logs")
        
        # SequentialChain Explanation
        st.write("### SequentialChain Architecture")
        st.write("""
        The system uses LangChain's SequentialChain to orchestrate the complete data processing pipeline. 
        SequentialChain allows multiple processing steps to be executed in sequence, with each step's 
        output becoming the input for the next step.
        """)
        
        st.code("""
SequentialChain(
    chains=[
        DataCleaningChain(),
        TextPreprocessingChain(),
        EmbeddingGenerationChain(),
        VectorStoreCreationChain()
    ],
    input_variables=["raw_data"],
    output_variables=["cleaned_data", "preprocessed_data", "embedded_data", "vector_store_path"],
    verbose=True
)
        """, language="python")
        
        # Chain Flow Diagram
        st.write("### Chain Processing Flow")
        st.code("""
Raw Data → DataCleaningChain → TextPreprocessingChain → EmbeddingGenerationChain → VectorStoreCreationChain
    ↓              ↓                      ↓                      ↓                      ↓
cleaned_data  preprocessed_data    embedded_data        vector_store_path      Final Results
        """, language="text")
        
        # Individual Chains
        st.write("### Individual Processing Chains")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔧 DataCleaningChain**")
            st.write("- Removes duplicates and missing values")
            st.write("- Cleans text descriptions")
            st.write("- Handles data validation")
            st.write("- **Input**: Raw book data")
            st.write("- **Output**: Cleaned data")
        
        with col2:
            st.write("**🔤 TextPreprocessingChain**")
            st.write("- Applies NLTK preprocessing")
            st.write("- Tokenization and lemmatization")
            st.write("- Stop word removal")
            st.write("- **Input**: Cleaned data")
            st.write("- **Output**: Preprocessed text")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**🧠 EmbeddingGenerationChain**")
            st.write("- Generates OpenAI embeddings")
            st.write("- Batch processing optimization")
            st.write("- Error handling and retries")
            st.write("- **Input**: Preprocessed data")
            st.write("- **Output**: Embedded vectors")
        
        with col4:
            st.write("**🗄️ VectorStoreCreationChain**")
            st.write("- Creates FAISS vector store")
            st.write("- Indexes embeddings")
            st.write("- Stores metadata mapping")
            st.write("- **Input**: Embedded data")
            st.write("- **Output**: Vector store path")
        
        # Base Chain Features
        st.write("### Base Chain Features")
        st.write("""
        All chains inherit from `MonitoredDataChain`, which provides:
        """)
        
        st.code("""
class MonitoredDataChain(Chain):
    def _call(self, inputs):
        start_time = time.time()
        result = self.execute_chain(inputs)
        
        # Built-in monitoring
        self.stats = {
            'execution_time': end_time - start_time,
            'input_size': len(inputs),
            'output_size': len(result),
            'timestamp': datetime.now().isoformat()
        }
        
        self.log_execution()
        return result
        """, language="python")
        
        # Monitoring and Statistics
        st.write("### Built-in Monitoring & Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Execution Metrics:**")
            st.write("- **Execution Time**: Performance tracking")
            st.write("- **Input Size**: Data volume monitoring")
            st.write("- **Output Size**: Result validation")
            st.write("- **Timestamps**: Process timing")
        
        with col2:
            st.write("**Logging Features:**")
            st.write("- **Chain Names**: Clear identification")
            st.write("- **Error Logging**: Failure tracking")
            st.write("- **Progress Updates**: Real-time status")
            st.write("- **Statistics Output**: Performance reports")
        
        # Example Output
        st.write("### Example Chain Execution Output")
        st.code("""
Chain 'DataCleaning' completed:
  Execution time: 2.34s
  Input size: 1
  Output size: 1

Chain 'TextPreprocessing' completed:
  Execution time: 15.67s
  Input size: 1
  Output size: 1

Chain 'EmbeddingGeneration' completed:
  Execution time: 45.23s
  Input size: 1
  Output size: 1

Chain 'VectorStoreCreation' completed:
  Execution time: 8.91s
  Input size: 1
  Output size: 1
        """, language="text")
        
        # Extensibility
        st.write("### Extending the Chain System")
        st.write("""
        The modular chain architecture makes it easy to add new processing steps:
        """)
        
        st.code("""
class CustomChain(MonitoredDataChain):
    def execute_chain(self, inputs):
        # Your custom processing logic
        data = inputs['input_data']
        processed_data = self.custom_process(data)
        return {'output_data': processed_data}
    
    def custom_process(self, data):
        # Implement your processing logic
        pass
        """, language="python")
        
        st.write("**Adding to Pipeline:**")
        st.code("""
# Add your custom chain to the pipeline
chains = [
    DataCleaningChain(),
    TextPreprocessingChain(),
    CustomChain(),  # Your new chain
    EmbeddingGenerationChain(),
    VectorStoreCreationChain()
]
        """, language="python")
        
    elif toc == "📊 Data Pipeline":
        st.subheader("📊 Data Pipeline")
        
        st.write("""
        The data pipeline is implemented using LangChain's SequentialChain architecture, providing 
        automated, monitored, and extensible data processing. Each step is implemented as a 
        separate chain that can be tested, modified, or replaced independently.
        """)
        
        st.write("### 1. Data Collection")
        st.write("""
        The system uses a systematic approach to collect diverse book data from Google Books API:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Search Strategies:**")
            st.write("- Subject-based: Fiction, Science, History, etc.")
            st.write("- Author-based: Popular authors across genres")
            st.write("- Title-based: Key terms and topics")
        
        with col2:
            st.write("**Quality Filters:**")
            st.write("- Minimum description length (2048 chars)")
            st.write("- Duplicate prevention")
            st.write("- Language filtering (English)")
            st.write("- Rate limiting with exponential backoff")
        
        st.write("### 2. Data Cleaning (DataCleaningChain)")
        st.write("""
        Raw data undergoes several cleaning steps through the automated DataCleaningChain:
        """)
        st.write("- **Automated Duplicate Removal**: Based on title and description similarity")
        st.write("- **Missing Value Handling**: Intelligent handling of incomplete records")
        st.write("- **Text Normalization**: Whitespace and formatting standardization")
        st.write("- **Quality Filtering**: Removal of empty or invalid descriptions")
        st.write("- **Built-in Monitoring**: Execution time and data volume tracking")
        
        st.write("### 3. Text Preprocessing (TextPreprocessingChain)")
        st.write("""
        Advanced text preprocessing improves similarity search quality through the TextPreprocessingChain:
        """)
        st.write("- **NLTK Integration**: Tokenization, lemmatization, POS tagging")
        st.write("- **Configurable Pipeline**: Adjustable preprocessing parameters")
        st.write("- **Stop Word Removal**: Common + book-specific terms")
        st.write("- **Book-specific Cleaning**: Remove ISBNs, page counts, etc.")
        st.write("- **Performance Optimization**: Efficient batch processing")
        st.write("- **Execution Monitoring**: Real-time progress tracking")
        
        st.write("### 4. Embedding Generation (EmbeddingGenerationChain)")
        st.write("""
        The EmbeddingGenerationChain handles the creation of semantic vectors:
        """)
        st.write("- **OpenAI Integration**: text-embedding-3-small model")
        st.write("- **Batch Processing**: Optimized API usage (50 documents per batch)")
        st.write("- **Error Handling**: Automatic retries and fallback mechanisms")
        st.write("- **Memory Management**: Efficient handling of large datasets")
        st.write("- **Progress Tracking**: Real-time embedding generation status")
        
        st.write("### 5. Vector Store Creation (VectorStoreCreationChain)")
        st.write("""
        The VectorStoreCreationChain builds the searchable index:
        """)
        st.write("- **FAISS Integration**: High-performance similarity search")
        st.write("- **Index Optimization**: Optimized for book recommendation queries")
        st.write("- **Metadata Storage**: Book information mapping")
        st.write("- **Persistent Storage**: Index saved to disk for reuse")
        st.write("- **Validation**: Index integrity and performance verification")
        
    elif toc == "🔍 Recommendation Engine":
        st.subheader("🔍 Recommendation Engine")
        
        st.write("### Semantic Search Process")
        st.write("""
        The recommendation engine uses semantic similarity to find relevant books:
        """)
        
        st.write("1. **Query Processing**: User input is preprocessed and embedded")
        st.write("2. **Vector Search**: FAISS performs similarity search")
        st.write("3. **Score Calculation**: Cosine distance between query and book embeddings")
        st.write("4. **Result Ranking**: Books ranked by similarity score")
        st.write("5. **Filtering**: Optional minimum similarity threshold")
        
        st.write("### Similarity Scoring")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Raw FAISS Scores:**")
            st.write("- Cosine distance (0-2 range)")
            st.write("- Lower = more similar")
            st.write("- Direct vector comparison")
        
        with col2:
            st.write("**Normalized Scores:**")
            st.write("- Converted to 0-1 range")
            st.write("- Higher = more similar")
            st.write("- More intuitive interpretation")
        
        st.write("### Search Features")
        st.write("- **Flexible Queries**: Natural language input")
        st.write("- **Configurable Results**: Adjustable number of recommendations")
        st.write("- **Score Filtering**: Minimum similarity thresholds")
        st.write("- **Preprocessing Options**: Query preprocessing for better matching")
        
    elif toc == "🔤 Text Preprocessing":
        st.subheader("🔤 Text Preprocessing Pipeline")
        
        st.write("### Preprocessing Steps")
        
        st.write("#### 1. Text Normalization")
        st.code("""
Input: "This groundbreaking BOOK explores..."
→ Unicode normalization (NFKC)
→ Case normalization (lowercase)
→ Whitespace normalization
Output: "this groundbreaking book explores..."
        """, language="text")
        
        st.write("#### 2. Tokenization & Filtering")
        st.code("""
Input: "this groundbreaking book explores..."
→ Word tokenization
→ Length filtering (min 2 chars)
→ Stop word removal
→ Number filtering (optional)
Output: ["groundbreaking", "book", "explores"]
        """, language="text")
        
        st.write("#### 3. Lemmatization")
        st.code("""
Input: ["groundbreaking", "explores", "civilizations"]
→ POS tagging
→ WordNet lemmatization
Output: ["groundbreaking", "explore", "civilization"]
        """, language="text")
        
        st.write("### Book-Specific Cleaning")
        st.write("The system removes book metadata that doesn't contribute to semantic meaning:")
        st.write("- ISBN numbers and identifiers")
        st.write("- Page counts and format information")
        st.write("- Publication dates and edition info")
        st.write("- Publisher metadata")
        
        st.write("### Configuration Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Text Processing:**")
            st.write("- Remove stop words")
            st.write("- Use lemmatization")
            st.write("- Normalize case")
            st.write("- Remove punctuation")
        
        with col2:
            st.write("**Filtering:**")
            st.write("- Remove numbers")
            st.write("- Minimum word length")
            st.write("- Book-specific patterns")
            st.write("- Custom stop words")
        
    elif toc == "🗄️ Vector Store":
        st.subheader("🗄️ FAISS Vector Store")
        
        st.write("### What is FAISS?")
        st.write("""
        FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and 
        clustering of dense vectors. It's optimized for high-dimensional vectors like text embeddings.
        """)
        
        st.write("### Vector Store Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance:**")
            st.write("- Fast similarity search")
            st.write("- Scalable to millions of vectors")
            st.write("- Memory-efficient indexing")
            st.write("- GPU acceleration support")
        
        with col2:
            st.write("**Search Capabilities:**")
            st.write("- k-nearest neighbors")
            st.write("- Range search")
            st.write("- Approximate search")
            st.write("- Batch queries")
        
        st.write("### Embedding Model")
        st.write("**OpenAI text-embedding-3-small:**")
        st.write("- 1536-dimensional vectors")
        st.write("- Optimized for semantic similarity")
        st.write("- Fast inference and low cost")
        st.write("- High-quality semantic representations")
        
        st.write("### Storage Structure")
        st.code("""
book_index/
├── index.faiss          # Vector index
└── index.pkl           # Metadata mapping
        """, language="text")
        
    elif toc == "⚙️ Configuration":
        st.subheader("⚙️ System Configuration")
        
        st.write("### Environment Variables")
        st.code("""
GOOGLE_API_KEY=your_google_books_api_key
OPENAI_API_KEY=your_openai_api_key
        """, language="env")
        
        st.write("### API Rate Limiting")
        st.write("**Google Books API:**")
        st.write("- 2-second delay between requests")
        st.write("- Exponential backoff for 429 errors")
        st.write("- Maximum 3 retries per query")
        st.write("- 300-second maximum wait time")
        
        st.write("**OpenAI API:**")
        st.write("- Batch processing (50 documents)")
        st.write("- Automatic retry on failures")
        st.write("- Cost optimization with small model")
        
        st.write("### Data Quality Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Collection Filters:**")
            st.write("- Min description: 2048 chars")
            st.write("- Language: English only")
            st.write("- Print type: Books only")
            st.write("- Duplicate prevention")
        
        with col2:
            st.write("**Processing Settings:**")
            st.write("- Embedding model: text-embedding-3-small")
            st.write("- Batch size: 50 documents")
            st.write("- Vector dimensions: 1536")
            st.write("- Index type: FAISS IndexFlatIP")
        
    elif toc == "🚀 API Integration":
        st.subheader("🚀 API Integration")
        
        st.write("### Google Books API")
        st.write("**Endpoint:** `https://www.googleapis.com/books/v1/volumes`")
        
        st.write("**Search Parameters:**")
        st.code("""
{
    "q": "search_query",
    "maxResults": 40,
    "startIndex": 0,
    "key": "API_KEY",
    "printType": "books",
    "langRestrict": "en"
}
        """, language="json")
        
        st.write("**Response Processing:**")
        st.write("- Extract volume information")
        st.write("- Filter by description quality")
        st.write("- Store metadata and content")
        st.write("- Handle pagination")
        
        st.write("### OpenAI Embeddings API")
        st.write("**Model:** `text-embedding-3-small`")
        
        st.write("**Request Format:**")
        st.code("""
{
    "input": ["text1", "text2", ...],
    "model": "text-embedding-3-small"
}
        """, language="json")
        
        st.write("**Response:**")
        st.write("- 1536-dimensional vectors")
        st.write("- Normalized embeddings")
        st.write("- Batch processing support")
        
        st.write("### Error Handling")
        st.write("**Rate Limiting:**")
        st.write("- Exponential backoff strategy")
        st.write("- Request queuing")
        st.write("- Graceful degradation")
        
        st.write("**API Failures:**")
        st.write("- Automatic retries")
        st.write("- Error logging")
        st.write("- Fallback mechanisms")

def automated_chains_interface():
    """Interface for automated LangChain processing chains"""
    st.header("🔗 LangChain Processing Chains")
    
    st.write("""
    This page demonstrates automated data processing using LangChain's SequentialChain architecture.
    The system implements a modular pipeline with built-in monitoring, error handling, and performance tracking.
    Each processing step is implemented as an independent chain that can be tested, modified, or replaced.
    """)
    
    # Chain configuration
    st.subheader("⚙️ LangChain Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔗 SequentialChain Pipeline:**")
        st.write("- **DataCleaningChain**: Automated data cleaning")
        st.write("- **TextPreprocessingChain**: NLTK-based text processing")
        st.write("- **EmbeddingGenerationChain**: OpenAI embedding creation")
        st.write("- **VectorStoreCreationChain**: FAISS index building")
        st.write("- **MonitoredDataChain**: Base class with monitoring")
    
    with col2:
        st.write("**🚀 LangChain Features:**")
        st.write("- **Built-in Monitoring**: Execution time and statistics")
        st.write("- **Error Handling**: Graceful failure recovery")
        st.write("- **Batch Processing**: Efficient data handling")
        st.write("- **Chain Composition**: Modular pipeline design")
        st.write("- **Extensibility**: Easy to add new chains")
    
    # Chain execution options
    st.subheader("🚀 Execute LangChain Pipeline")
    
    st.write("Execute the complete data processing pipeline using LangChain's SequentialChain.")
    st.write("The pipeline will run all chains in sequence with built-in monitoring and error handling.")
    
    if st.button("🔄 Run LangChain Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running LangChain SequentialChain pipeline..."):
            try:
                results = run_data_pipeline(books_file)
                
                if results:
                    st.success("✅ LangChain SequentialChain pipeline completed successfully!")
                    
                    # Display metrics
                    st.subheader("📊 LangChain Pipeline Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Cleaned Records", len(results['cleaned_data']))
                    
                    with col2:
                        st.metric("Preprocessed Records", len(results['preprocessed_data']))
                    
                    with col3:
                        st.metric("Embedded Records", len(results['embedded_data']))
                    
                    with col4:
                        st.metric("Vector Store", "✅ Created")
                    
                    # Show sample data
                    with st.expander("📋 Sample Processed Data"):
                        sample_data = results['preprocessed_data'][['title', 'description_processed']].head()
                        st.dataframe(sample_data, use_container_width=True)
                    
                    # Show processing statistics
                    st.subheader("📈 LangChain Chain Statistics")
                    st.info("Check the terminal output for detailed LangChain chain execution times and statistics.")
                    st.write("Each chain provides execution metrics including timing, input/output sizes, and performance data.")
                    
                else:
                    st.warning("⚠️ Pipeline completed but no results returned.")
                    
            except Exception as e:
                st.error(f"❌ Error in LangChain pipeline: {str(e)}")
                st.info("💡 Check the console for detailed LangChain chain error information.")
    
    


def main():
    """Main application"""
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Dashboard"
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔗 Automated Chains", "🔍 Recommendations", "📚 Documentation"],
        index=["🏠 Dashboard", "🔗 Automated Chains", "🔍 Recommendations", "📚 Documentation"].index(st.session_state.page)
    )
    
    # Update session state when page changes
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Check environment variables
    google_api = "✅" if os.getenv("GOOGLE_API_KEY") else "❌"
    openai_api = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
    
    st.sidebar.write(f"Google API Key: {google_api}")
    st.sidebar.write(f"OpenAI API Key: {openai_api}")
    
    # Check data files
    data_exists = "✅" if os.path.exists("../data/books.csv") else "❌"
    embeddings_exists = "✅" if os.path.exists("../data/books_embeddings.csv") else "❌"
    vector_store_exists = "✅" if os.path.exists("../book_index") else "❌"
    
    st.sidebar.write(f"Raw Data: {data_exists}")
    st.sidebar.write(f"Embeddings: {embeddings_exists}")
    st.sidebar.write(f"Vector Store: {vector_store_exists}")
    
    # Page routing
    if st.session_state.page == "🏠 Dashboard":
        st.header("📊 System Dashboard")
        
        # Load data
        df = load_data()
        # System overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("System Overview")
            if df is not None:
                create_dashboard_metrics(df)
                create_visualizations(df)
            else:
                st.info("No data available. Please run the automated pipeline first.")
        
        with col2:
            st.subheader("Quick Actions")
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            if st.button("🔍 Try Recommendations"):
                st.session_state.page = "🔍 Recommendations"
                st.rerun()
            
            if st.button("🔗 Automated Chains"):
                st.session_state.page = "🔗 Automated Chains"
                st.rerun()
    
    elif st.session_state.page == "🔍 Recommendations":
        recommendation_interface()
    
    elif st.session_state.page == "🔗 Automated Chains":
        automated_chains_interface()
    
    elif st.session_state.page == "📚 Documentation":
        documentation_page()

if __name__ == "__main__":
    main()
