"""Test-only compatibility adapter for the current Starlette/httpx stack.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

import asyncio

import httpx


class SyncASGIClient:
    """Test-only synchronous adapter that runs requests against a FastAPI app through
    httpx's ASGI transport, standing in for fastapi.testclient.TestClient on the current
    Starlette/httpx stack."""
    def __init__(self, app, **kwargs):
        self.app = app

    def request(self, method, url, **kwargs):
        """Runs the given HTTP method/URL against the wrapped ASGI app over
        httpx.ASGITransport, blocking via asyncio.run, and returns the response."""
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def pytest_configure():
    """Replaces fastapi.testclient.TestClient with SyncASGIClient for the duration of
    the test session."""
    import fastapi.testclient

    fastapi.testclient.TestClient = SyncASGIClient
