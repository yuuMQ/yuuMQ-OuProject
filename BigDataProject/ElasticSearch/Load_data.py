import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


es = Elasticsearch("http://localhost:9200", headers={"Content-Type": "application/json"})
index_name = "search_index"
file_path = "../OuProject/ou_data.json"
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    logger.error(f"File not found at {file_path}")
    exit()

if es.ping():
    logger.info("Successfully connected to Elasticsearch")
else:
    logger.info("Failed to connect to Elasticsearch")

batch_size = 75000

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    actions = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": i + idx,
            "_source": doc
        }
        for idx, doc in enumerate(batch)
    ]
    try:
        success, failed = bulk(es, actions, refresh=False)
        logger.info(f"Batch {i // batch_size + 1}: Successfully indexed {success} documents, failed {failed}.")
    except Exception as e:
        logger.error(f"Error occurred while indexing batch {i // batch_size + 1}: {e}")

es.transport.close()
