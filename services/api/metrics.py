from prometheus_client import Counter, Histogram


HTTP_REQUESTS_COUNTER = Counter(
    name="http_requests_total",
    documentation="Total number of HTTP requests",
    labelnames=["method", "route", "status"]
)


HTTP_DURATION_HISTOGRAM = Histogram(
    name="http_requests_duration_seconds",
    documentation="Duration of HTTP requests",
    labelnames=["method", "route"],
    buckets=[
        0.001,
        0.010,
        0.025,
        0.050,
        0.100,
        0.250,
        0.500,
        1.000,
        2.000,
        5.000,
        10.000
    ]
)
