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
    """Concise documentation page for the Book Recommendation System"""
    st.header("📚 System Documentation")

    toc = st.sidebar.radio(
        "Jump to section:",
        [
            "System Overview",
            "How It Works",
            "Main Features",
            "Configuration",
            "How to Use",
            "Key Technologies"
        ]
    )

    if toc == "System Overview":
        st.subheader("System Overview")
        st.write("""
        The Book Recommendation System is a modular application that enables users to discover books using semantic search. It integrates data collection, processing, and recommendation into a single, user-friendly interface.
        """)
        st.write("**Main Components:**")
        st.markdown("- Data Collection (Google Books API)\n- Data Processing Pipeline (LangChain)\n- Embedding & Vector Store (OpenAI, FAISS)\n- Recommendation Engine (Semantic Search)\n- Streamlit User Interface")

    elif toc == "How It Works":
        st.subheader("How It Works")
        st.write("""
        1. **Data Collection:** Gathers book data from the Google Books API.
        2. **Processing Pipeline:** Cleans and preprocesses text, generates embeddings, and builds a searchable vector store.
        3. **Recommendation:** User queries are semantically matched to books using the vector store.
        """)
        st.code("""
Google Books API → Data Cleaning → Text Preprocessing → Embeddings → FAISS Vector Store → Recommendations
        """, language="text")

    elif toc == "Main Features":
        st.subheader("Main Features")
        st.write("""
        - **Dashboard:** Visualizes book data, categories, and authors.
        - **Automated Chains:** Runs the full data processing pipeline with monitoring and error handling.
        - **Recommendations:** Provides semantic book search with adjustable options.
        """)
        st.write("Navigate using the sidebar to access each feature.")

    elif toc == "Configuration":
        st.subheader("Configuration")
        st.write("""
        - **API Keys:** Set the following environment variables:
        """)
        st.code("""
GOOGLE_API_KEY=your_google_books_api_key
OPENAI_API_KEY=your_openai_api_key
        """, language="env")
        st.write("""
        - **Data Files:**
            - Raw data: `data/books.csv`
            - Embeddings: `data/books_embeddings.csv`
            - Vector store: `book_index/`
        - **Index Files:**
            - `book_index/index.faiss` (vector index)
            - `book_index/index.pkl` (metadata)
        """)

    elif toc == "How to Use":
        st.subheader("How to Use")
        st.write("""
        1. **Run the Automated Pipeline:**
            - Go to 'Automated Chains' and click 'Run LangChain Pipeline' to process data and build the vector store.
        2. **Get Recommendations:**
            - Go to 'Recommendations', enter a query, and receive book suggestions.
        3. **Dashboard:**
            - View data metrics and visualizations on the Dashboard.
        """)
        st.write("**Common Commands:**")
        st.code("""
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run src/interface.py
        """, language="bash")
        st.write("**Troubleshooting:**")
        st.markdown("- Ensure API keys are set.\n- Check that data and index files exist.\n- If recommendations fail, rerun the pipeline.")

    elif toc == "Key Technologies":
        st.subheader("Key Technologies")
        st.write("""
        - **Streamlit:** User interface
        - **Pandas:** Data manipulation
        - **Plotly:** Visualizations
        - **LangChain:** Data processing pipeline
        - **OpenAI:** Embedding generation
        - **FAISS:** Vector similarity search
        - **NLTK:** Text preprocessing
        """)

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
