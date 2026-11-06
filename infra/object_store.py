from hashlib import sha256
from io import BytesIO
import os

from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager, Timeout


class OriginalStore:
    def __init__(self) -> None:
        self.bucket = os.getenv("RAG_OBJECT_BUCKET", "rag-python-documents")
        self.http = PoolManager(timeout=Timeout(connect=6, read=30), retries=False)
        self.client = Minio(
            os.getenv("RAG_OBJECT_ENDPOINT", "127.0.0.1:19000"),
            access_key=os.environ["RAG_OBJECT_ACCESS_KEY"],
            secret_key=os.environ["RAG_OBJECT_SECRET_KEY"],
            secure=os.getenv("RAG_OBJECT_TLS", "false").lower() == "true",
            http_client=self.http,
        )

    def initialize(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as exc:
                if exc.code != "BucketAlreadyOwnedByYou":
                    raise

    def close(self) -> None:
        self.http.clear()

    def put(self, document_id: str, data: bytes, media_type: str) -> None:
        self.client.put_object(
            self.bucket, f"documents/{document_id}/original", BytesIO(data), len(data),
            content_type=media_type,
        )

    def get(self, document_id: str, expected_digest: str, size: int) -> bytes:
        response = self.client.get_object(self.bucket, f"documents/{document_id}/original")
        try:
            data = response.read(size + 1)
        finally:
            response.close()
            response.release_conn()
        if len(data) != size or sha256(data).hexdigest() != expected_digest:
            raise OSError("Stored original failed integrity verification")
        return data
