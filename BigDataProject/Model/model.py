from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import torch
from ElasticSearch.Get_data import get_data_from_es

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tfidf_model(documents):
    documents_normalized = [
        {"url": doc["url"],
         "title": unidecode(doc["title"]),
         "content": unidecode(doc["content"])}
        for doc in documents
    ]
    with open("./stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()

    vectorizer_title = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), stop_words=stopwords)
    vectorizer_content = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), stop_words=stopwords)
    tfidf_matrix_title = vectorizer_title.fit_transform([doc["title"] for doc in documents_normalized])
    tfidf_matrix_content = vectorizer_content.fit_transform([doc["content"] for doc in documents_normalized])

    return tfidf_matrix_title, tfidf_matrix_content, vectorizer_title, vectorizer_content

def search_tfidf(query, vectorizer_title, vectorizer_content, tfidf_matrix_title, tfidf_matrix_content, documents):
    if not query.strip():
        return []

    query_normalized = unidecode(query)

    query_vec_title = vectorizer_title.transform([query_normalized])
    query_vec_content = vectorizer_content.transform([query_normalized])

    similarity_title = cosine_similarity(query_vec_title, tfidf_matrix_title)
    similarity_content = cosine_similarity(query_vec_content, tfidf_matrix_content)

    similarity = (similarity_title + similarity_content) / 2
    results = []
    for idx, score in enumerate(similarity[0]):
        if score > 0:
            results.append({
                "url": documents[idx]["url"],
                "title": documents[idx]["title"],
                "content": documents[idx]["content"],
                "score": score,
            })

    results = sorted(results, key=lambda d: d["score"], reverse=True)
    return results


def create_bert_model(documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    documents_text = [doc["title"] + " " + doc["content"] for doc in documents]
    embeddings = model.encode(documents_text, convert_to_tensor=True, show_progress_bar=True)
    return embeddings, model

def search_bert(query, model, embeddings, documents):
    query_vec = model.encode([query], convert_to_tensor=True).to(device)
    similarities = cosine_similarity(query_vec.cpu().numpy(), embeddings.cpu().numpy())
    results = []
    for idx, score in enumerate(similarities[0]):
        if score > 0:
            results.append({
                "url": documents[idx]["url"],
                "title": documents[idx]["title"],
                "content": documents[idx]["content"],
                "score": score,
            })
    results = sorted(results, key=lambda d: d["score"], reverse=True)
    return results

def search_combined(query,
                    vectorizer_title, vectorizer_content, tfidf_matrix_title, tfidf_matrix_content,
                    model, embeddings, documents,
                    tfidf_weight=0.6, bert_weight=0.4,
                    ):
    results_tfidf = search_tfidf(query, vectorizer_title, vectorizer_content,
                                 tfidf_matrix_title, tfidf_matrix_content, documents)
    results_bert = search_bert(query, model, embeddings, documents)

    combined_results = {}
    for result in results_tfidf:
        result["score"] *= tfidf_weight
        if result["url"] not in combined_results:
            combined_results[result["url"]] = result
        else:
            combined_results[result["url"]]["score"] = max(combined_results[result["url"]]["score"], result["score"])
    for result in results_bert:
        result["score"] *= bert_weight
        if result["url"] not in combined_results:
            combined_results[result["url"]] = result
        else:
            combined_results[result["url"]]["score"] = max(combined_results[result["url"]]["score"], result["score"])

    unique_results = sorted(combined_results.values(), key=lambda d: d["score"], reverse=True)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unique_results

if __name__ == '__main__':
    documents = get_data_from_es()
    tfidf_matrix_title, tfidf_matrix_content, vectorizer_title, vectorizer_content = create_tfidf_model(documents)
    embeddings, model = create_bert_model(documents)

    while(1):
        query = input("Nhập câu hỏi: ")
        results = search_combined(query, vectorizer_title, vectorizer_content, tfidf_matrix_title, tfidf_matrix_content,
                                  model, embeddings, documents)

        for idx, res in enumerate(results[:5]):
            print("Câu trả lời:")
            print(f"{idx + 1}. [{res['title']}]({res['url']}) - Score: {res['score']:.4f}")