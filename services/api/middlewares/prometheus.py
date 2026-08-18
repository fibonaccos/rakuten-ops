from fastapi import Request
from time import perf_counter

from metrics import HTTP_DURATION_HISTOGRAM, HTTP_REQUESTS_COUNTER


async def prometheus_middlware(request: Request, call_next):
    if request.url.path in ("/metrics", "/health", "/ready"):
        return await call_next(request)

    status = "500"
    start = perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        duration = perf_counter() - start

        route = request.scope.get("route")
        route_path = route.path if route else request.url.path

        HTTP_REQUESTS_COUNTER.labels(
            method=request.method,
            route=route_path,
            status=status,
        ).inc()
        HTTP_DURATION_HISTOGRAM.labels(
            method=request.method,
            route=route_path,
        ).observe(duration)
