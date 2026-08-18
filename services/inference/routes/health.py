from fastapi import APIRouter

from _version import __version__


router = APIRouter(prefix="", tags=["Defaults"])


@router.get("/health", summary="health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "API is ready to use.",
        "api_version": __version__  # TODO: Inject the docker image tag for the current version.
    }
