import qdrant_client
import logging
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)

QDRANT_URL = "https://8fe96bbe-5d1a-4be8-a5b4-7c93cccab7e8.us-east-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjA5YTI2NGUtN2E0Ni00YTBhLTlhNzAtYTExNmI1M2QyYjA5In0.FlIEcza_AEzzmJALdqnSxYmGQ2M1Et7Bj0txqOyftCc"
COLLECTION_NAME = "msmarco_chunks"

cloud_client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
local_client = qdrant_client.QdrantClient(path="./local_qdrant_db")

# Get all points
print("Fetching from cloud...")
points = cloud_client.scroll(
    collection_name=COLLECTION_NAME,
    limit=10000,
    with_payload=True,
    with_vectors=True,
)[0]

print(f"Fetched {len(points)} points. Creating local collection...")

if not local_client.collection_exists(COLLECTION_NAME):
    local_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

from qdrant_client.models import PointStruct
local_points = []
for p in points:
    local_points.append(PointStruct(id=p.id, vector=p.vector, payload=p.payload))

local_client.upsert(
    collection_name=COLLECTION_NAME,
    points=local_points
)

print("Successfully migrated to local Qdrant DB!")
