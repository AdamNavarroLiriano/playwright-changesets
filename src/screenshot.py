from playwright.sync_api import Playwright, sync_playwright


def run(
    playwright: Playwright, url: str, selector_wait: str, path: str
) -> tuple[str, bool]:
    """
    Captures a screenshot of a webpage after waiting for a specific element to appear.

    Args:
        playwright (Playwright): The Playwright instance used to control the browser.
        url (str): The URL of the webpage to navigate to.
        selector_wait (str): The CSS selector to wait for before taking the screenshot.
        path (str): The file path where the screenshot will be saved.

    Returns:
        tuple[str, bool]: A tuple containing the URL and a boolean indicating success (True) or failure (False).

    Raises:
        Exception: If an error occurs during the process, it is caught and printed.
    """
    try:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector(selector_wait)
        page.screenshot(path=path)
        browser.close()

        return (url, True)

    except Exception as e:
        print(f"Error: {e}")
        return (url, False)


url = "https://overpass-api.de/achavi/?changeset=148420905"
selector = "#OpenLayers_Layer_Vector_113_root"
path = "../data/148420905.png"

# from playwright.async_api import async_playwright

# playwright = await async_playwright().start()
# browser = await playwright.chromium.launch(headless=True)
# page = await browser.new_page()

# await page.goto("https://overpass-api.de/achavi/?changeset=142835204")
# # await page.locator("xpath=/html/body/div[1]").screenshot(path="screenshot.png")


# await page.wait_for_selector("#OpenLayers_Layer_Vector_113_root")

# # Take a screenshot of the element
# element = await page.query_selector("#map_div")
# await element.screenshot(path="map_div_screenshot.png")

# await page.screenshot(path="screenshot.png", full_page=True)

# await browser.close()

with sync_playwright() as playwright:
    run(playwright, url, selector_wait=selector, path=path)
