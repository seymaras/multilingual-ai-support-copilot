from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI


def _package_version() -> str:
    try:
        return version("multilingual-ai-support-copilot")
    except PackageNotFoundError:
        return "0.0.0"


app = FastAPI(
    title="Multilingual AI Support Copilot",
    version=_package_version(),
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
