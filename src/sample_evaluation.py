import os
import pandas as pd
from datetime import datetime
from core.text_preprocessor import preprocessing_pipeline
from services.recommendation_service import recommend_books

# Sample queries for evaluation
eval_queries = [
    "machine learning and artificial intelligence",
    "history of ancient civilizations",
    "psychology of motivation",
    "modern poetry",
    "business leadership",
    "climate change and sustainability",
    "children's literature",
    "space exploration",
    "biography of famous scientists",
    "art and creativity"
]

# Number of recommendations to fetch for each query
TOP_K = 5

# Load the books data for reference
books_path = os.path.join("..", "data", "books.csv")
if os.path.exists(books_path):
    books_df = pd.read_csv(books_path)
else:
    books_df = None

# Preprocessing pipeline
preprocessor = preprocessing_pipeline()

def evaluate_recommendation_pipeline(use_preprocessing=True):
    print(f"\n=== Book Recommendation Pipeline Evaluation (use_preprocessing={use_preprocessing}) ===\n")
    all_results = []
    for query in eval_queries:
        processed_query = preprocessor.preprocess_text(query) if use_preprocessing else query
        print(f"\nQuery: '{query}' (preprocessed: '{processed_query}' if use_preprocessing else 'raw')")
        try:
            results = recommend_books(processed_query, TOP_K)
            if results is None or results.empty:
                print("  No recommendations found.")
                continue
            print(f"  Top {TOP_K} recommendations:")
            for i, row in results.iterrows():
                print(f"    {i+1}. {row['title']} by {row['authors']} (Score: {row['similarity_score']:.3f})")
            all_results.append((query, results))
        except Exception as e:
            print(f"  Error: {e}")
    return all_results

def analyze_results(all_results):
    print("\n=== Analysis of Recommendation Results ===\n")
    # Aggregate statistics
    total_recs = 0
    unique_titles = set()
    score_list = []
    for query, df in all_results:
        total_recs += len(df)
        unique_titles.update(df['title'].tolist())
        score_list.extend(df['similarity_score'].tolist())
    print(f"Total queries evaluated: {len(all_results)}")
    print(f"Total recommendations returned: {total_recs}")
    print(f"Unique recommended titles: {len(unique_titles)}")
    if score_list:
        print(f"Average similarity score: {sum(score_list)/len(score_list):.3f}")
        print(f"Max similarity score: {max(score_list):.3f}")
        print(f"Min similarity score: {min(score_list):.3f}")
    # Optionally, analyze category/author diversity, etc.
    top_categories = None
    top_authors = None
    if books_df is not None:
        rec_titles = set(unique_titles)
        rec_books = books_df[books_df['title'].isin(rec_titles)]
        if not rec_books.empty:
            top_categories = rec_books['primary_category'].value_counts().head(5)
            print("Top recommended categories:")
            for cat, count in top_categories.items():
                print(f"  {cat}: {count}")
            top_authors = rec_books['authors'].value_counts().head(5)
            print("Top recommended authors:")
            for author, count in top_authors.items():
                print(f"  {author}: {count}")
    return {
        'total_queries': len(all_results),
        'total_recs': total_recs,
        'unique_titles': len(unique_titles),
        'avg_score': sum(score_list)/len(score_list) if score_list else 0,
        'max_score': max(score_list) if score_list else 0,
        'min_score': min(score_list) if score_list else 0,
        'top_categories': top_categories,
        'top_authors': top_authors
    }

def generate_report(metrics, save_path=None):
    print("\n=== Recommendation System Performance Report ===\n")
    lines = []
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total queries evaluated: {metrics['total_queries']}")
    lines.append(f"Total recommendations returned: {metrics['total_recs']}")
    lines.append(f"Unique recommended titles: {metrics['unique_titles']}")
    lines.append(f"Average similarity score: {metrics['avg_score']:.3f}")
    lines.append(f"Max similarity score: {metrics['max_score']:.3f}")
    lines.append(f"Min similarity score: {metrics['min_score']:.3f}")
    if metrics['top_categories'] is not None:
        lines.append("Top recommended categories:")
        for cat, count in metrics['top_categories'].items():
            lines.append(f"  {cat}: {count}")
    if metrics['top_authors'] is not None:
        lines.append("Top recommended authors:")
        for author, count in metrics['top_authors'].items():
            lines.append(f"  {author}: {count}")
    lines.append("")
    # Suggestions for improvement
    lines.append("=== Suggestions for Improvement ===")
    if metrics['avg_score'] < 0.5:
        lines.append("- Improve the quality of book descriptions and metadata for better semantic matching.")
    if metrics['unique_titles'] < metrics['total_recs'] * 0.7:
        lines.append("- Increase diversity in recommendations by tuning the similarity threshold or using additional features.")
    if metrics['top_categories'] is not None and metrics['top_categories'].max() > metrics['total_recs'] * 0.3:
        lines.append("- Recommendations are concentrated in a few categories. Consider expanding the dataset or search strategies for more variety.")
    if metrics['avg_score'] > 0.7:
        lines.append("- The system is performing well. Consider scaling up the dataset or adding more advanced models for further improvement.")
    if len(lines) == 8:  # Only the header and stats, no specific suggestions
        lines.append("- System is performing well. No major issues detected.")
    report = "\n".join(lines)
    print(report)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {save_path}")
    return report

def main():
    # With NLTK preprocessing
    all_results_pre = evaluate_recommendation_pipeline(use_preprocessing=True)
    metrics_pre = analyze_results(all_results_pre)
    print("\n---\n")
    # Without NLTK preprocessing
    all_results_raw = evaluate_recommendation_pipeline(use_preprocessing=False)
    metrics_raw = analyze_results(all_results_raw)
    print("\n=== Comparison of Recommendation Performance ===\n")
    print(f"With NLTK preprocessing: Avg score = {metrics_pre['avg_score']:.3f}, Unique titles = {metrics_pre['unique_titles']}")
    print(f"Without NLTK preprocessing: Avg score = {metrics_raw['avg_score']:.3f}, Unique titles = {metrics_raw['unique_titles']}")
    # Save both reports
    generate_report(metrics_pre, save_path="recommendation_performance_report_nltk.txt")
    generate_report(metrics_raw, save_path="recommendation_performance_report_raw.txt")
    print("\nEvaluation complete. See 'recommendation_performance_report_nltk.txt' and 'recommendation_performance_report_raw.txt' for details.\n")

if __name__ == "__main__":
    main() 