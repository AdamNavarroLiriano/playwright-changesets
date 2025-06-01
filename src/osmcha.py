import asyncio

import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINTS: tuple[str, ...] = (
    "aoi",
    "blacklisted-users",
    "challenges",
    "changesets",
    "features",
    "mapping-team",
    "social-auth",
    "stats",
    "suspicion-reasons",
    "tags",
    "update-deleted-users",
    "user-stats",
    "users",
    "whitelist-user",
)


class OSMChaAPI:
    def __init__(self, token: str, base_url="https://osmcha.org/api/v1/"):
        self.token = token
        self.base_url = base_url

    def __repr__(self) -> str:
        return f"OSMChaAPI(token=*********, base_url={self.base_url})"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"{self.token}",
        }

    def create_request_path(self, endpoint: str, **kwargs) -> str:
        if endpoint not in ENDPOINTS:
            raise ValueError(
                f"Invalid endpoint: {endpoint}. Must be one of {ENDPOINTS}."
            )

        base_request_url = f"{self.base_url}{endpoint}"

        # If no parameters are provided, return the base URL for the endpoint
        if not kwargs:
            return base_request_url

        params = "&".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{base_request_url}?{params}"

    async def get(self, endpoint: str, **kwargs) -> dict:
        """
        Make a GET request to the OSMCha API.
        """
        url = self.create_request_path(endpoint, **kwargs)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise requests.HTTPError(
                        f"Request failed with status {response.status}"
                    )
                return await response.json()


if __name__ == "__main__":
    import os

    token: str = os.environ["OSMCHA_APIKEY"]

    osmcha_api = OSMChaAPI(token)

    suspicion_reasons = await osmcha_api.get("suspicion-reasons", page_size=1_000)
    tags = await osmcha_api.get("tags", page_size=1_000)

    harmful_changesets = await osmcha_api.get(
        "changesets",
        checked=True,
        harmful=True,
        page_size=1_000,
        number_reasons__gte=3,
        date_gte="2025-04-01",
        date_lte="2025-04-30",
    )
