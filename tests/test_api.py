from importlib.metadata import version

from fastapi.testclient import TestClient

from multilingual_support_copilot.api import app


def test_health_returns_ok_without_loading_models() -> None:
    # The autouse fixture makes any real model/LLM getter raise, so a 200
    # here also proves the health check does not touch them.
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_version_comes_from_package_metadata() -> None:
    assert app.version == version("multilingual-ai-support-copilot")
