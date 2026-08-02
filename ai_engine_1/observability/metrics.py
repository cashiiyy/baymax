import time
from typing import Dict, Any
from pydantic import BaseModel

class SystemMetrics(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_llm_latency_ms: float = 0.0
    avg_embedding_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    cache_hit_count: int = 0
    gpu_utilization_percent: float = 0.0
    memory_usage_mb: float = 0.0

class MetricsCollector:
    """Tracks latency, request throughput, and cache metrics for AI Engine 1."""

    def __init__(self):
        self.metrics = SystemMetrics()
        self._llm_latencies = []
        self._embedding_latencies = []
        self._retrieval_latencies = []

    def record_request(self, success: bool = True):
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

    def record_llm_latency(self, latency_ms: float):
        self._llm_latencies.append(latency_ms)
        self.metrics.avg_llm_latency_ms = round(sum(self._llm_latencies) / len(self._llm_latencies), 2)

    def record_embedding_latency(self, latency_ms: float):
        self._embedding_latencies.append(latency_ms)
        self.metrics.avg_embedding_latency_ms = round(sum(self._embedding_latencies) / len(self._embedding_latencies), 2)

    def get_summary(self) -> Dict[str, Any]:
        return self.metrics.dict()

metrics_collector = MetricsCollector()
