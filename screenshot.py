import re

from playwright.sync_api import sync_playwright


def run(playwright, url, take_screenshot):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto(url)

    if take_screenshot:
        __capture_screenshot(page, url)
    else:
        __save_page_text(page, "main", url)

    browser.close()


def __save_page_text(page, selector, url):
    title = page.title()
    main_content = page.query_selector(selector)
    main_text = (
        main_content.inner_text() if main_content else "No requested selector found"
    )

    filename = __safe_filename_from(title)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Title: {title}\n\n")
        f.write(main_text)

    print(f"Data saved as {filename}")


def __safe_filename_from(title):
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return f"{safe_title}.txt"


def __capture_screenshot(page, url):
    filename = __safe_filename_from(page.title()) + ".png"
    page.screenshot(path=filename, full_page=True)
    print(f"Screenshot saved as {filename}")


from playwright.async_api import async_playwright

playwright = await async_playwright().start()
browser = await playwright.chromium.launch(headless=True)
page = await browser.new_page()

await page.goto("https://overpass-api.de/achavi/?changeset=142782344")
# await page.locator("xpath=/html/body/div[1]").screenshot(path="screenshot.png")

await page.wait_for_selector("#map_div")

await page.wait_for_selector(".map-loaded-indicator")

# Take a screenshot of the element
element = await page.query_selector("#map_div")
await element.screenshot(path="map_div_screenshot.png")

await page.screenshot(path="screenshot.png", full_page=True)


with sync_playwright() as playwright:
    run(playwright, url, take_screenshot)
