from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", headers={"Content-Type": "application/json"})
index_name = "search_index"

def create_index():
    # Xoa index neu da ton tai
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"🗑️ Deleted existing index: {index_name}")

    # Cau hinh index_mapping
    index_mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "url": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "content": {"type": "text", "analyzer": "standard"},
                "hyperlink_level": {"type": "integer"}
            }
        }
    }

    # Tao chi muc
    es.indices.create(index=index_name, settings=index_mapping["settings"], mappings=index_mapping["mappings"])
    print(f"Created index: {index_name}")

if __name__ == '__main__':
    create_index()