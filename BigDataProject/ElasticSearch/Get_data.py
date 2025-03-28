from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", headers={"Content-Type": "application/json"})
index_name = "search_index"
def get_data_from_es():
    query = {
        "query":{
            "match_all": {}
        }
    }
    response = es.search(index=index_name, body=query, size=10000)
    return[{
        "url": hit["_source"]["url"],
        "title": hit["_source"]["title"],
        "content": hit["_source"]["content"],
    } for hit in response["hits"]["hits"]]
