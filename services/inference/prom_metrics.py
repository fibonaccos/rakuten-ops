from prometheus_client import Counter, Histogram


INFERENCE_REQUESTS = Counter(
    name="inference_requests_total",
    documentation="Total number of inference requests",
    labelnames=["mode", "model", "status"]
)


INFERENCE_DURATION = Histogram(
    name="inference_duration_seconds",
    documentation="Model inference duration",
    labelnames=["mode", "model"]
)
