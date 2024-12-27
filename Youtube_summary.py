from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def fetch_subtitle(link):
    """Fetch subtitles from a YouTube video link."""
    unique_id = link.split("=")[-1]
    sub = YouTubeTranscriptApi.get_transcript(unique_id)  
    return " ".join([x['text'] for x in sub])

def cluster_sentences(sentences, n_clusters=5):
    """Cluster sentences into thematic groups."""
    tf_idf = TfidfVectorizer(stop_words='english')
    sentence_vectors = tf_idf.fit_transform(sentences)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(sentence_vectors)
    return clusters

def summarize_clusters(sentences, clusters):
    """Summarize each cluster."""
    summaries = []
    for cluster_id in set(clusters):
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if clusters[i] == cluster_id]
        
        # Summarize each cluster using TF-IDF
        tf_idf = TfidfVectorizer(stop_words='english')
        sentence_vectors = tf_idf.fit_transform(cluster_sentences)
        sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
        
        # Get top sentence from the cluster
        top_sentence_index = np.argsort(sent_scores, axis=0)[::-1][0]
        summaries.append(cluster_sentences[top_sentence_index])
    
    return summaries

def main():
    link = "https://www.youtube.com/watch?v=o78Kh3me3fY"  # Example link
    subtitle = fetch_subtitle(link)
    sentences = sent_tokenize(subtitle)
    
    # Cluster sentences
    n_clusters = 5  # Adjust based on video length/topics
    clusters = cluster_sentences(sentences, n_clusters=n_clusters)
    
    # Summarize each cluster
    summaries = summarize_clusters(sentences, clusters)
    
    # Print section-wise summaries
    for i, summary in enumerate(summaries, 1):
        print(f"Section {i}:")
        print(summary)
        print("-" * 40)

if __name__ == "__main__":
    main()