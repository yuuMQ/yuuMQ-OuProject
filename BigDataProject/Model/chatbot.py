import openai
import torch
from ElasticSearch.Get_data import get_data_from_es
from model import create_tfidf_model, create_bert_model, search_combined




# OpenAI API Key
openai.api_key = "YOUR API KEY HERE"
# Xác định thiết bị (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Đang tải dữ liệu từ Elasticsearch...")
documents = get_data_from_es()
print("🔄 Đang tạo mô hình TF-IDF và BERT...")
tfidf_matrix_title, tfidf_matrix_content, vectorizer_title, vectorizer_content = create_tfidf_model(documents)
embeddings, model = create_bert_model(documents)


def search_info(query):
    """
    Tìm kiếm thông tin liên quan bằng TF-IDF + BERT
    """
    results = search_combined(query, vectorizer_title, vectorizer_content,
                              tfidf_matrix_title, tfidf_matrix_content,
                              model, embeddings, documents)

    if results:
        best_result = results[0]  # Lấy kết quả tốt nhất
        context = best_result["title"] + "\n" + best_result["content"]
    else:
        context = ""

    return context


def generate_response_with_gpt(query, context=""):
    """
    Gửi query đến GPT-3.5-turbo, nếu có context thì sử dụng để trả lời tốt hơn.
    """
    prompt = f"""
    Bạn là một trợ lý AI. Hãy trả lời câu hỏi dựa trên thông tin sau nếu có:

    Câu hỏi: {query}

    {("Thông tin liên quan:\n" + context) if context else ""}
    """
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Bạn là một trợ lý AI."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def chatbot():
    """
    Vòng lặp chatbot: nhập câu hỏi từ terminal và nhận câu trả lời từ GPT.
    """
    print("🚀 Chatbot AI (Nhập 'exit' để thoát)\n")
    while True:
        query = input("💬 Nhập câu hỏi: ").strip()
        if query.lower() == "exit":
            print("👋 Tạm biệt!")
            break

        # 🔎 Tìm kiếm thông tin bằng TF-IDF + BERT
        context = search_info(query)

        # 🤖 GPT trả lời dựa trên thông tin tìm được
        response = generate_response_with_gpt(query, context)
        print("\n📌 Chatbot Response:\n", response)


if __name__ == "__main__":
    chatbot()
