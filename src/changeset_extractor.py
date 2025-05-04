import ast
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


# Thanks to https://gist.github.com/kumaraditya303/e6dee949dda298b35d167369955d45c6
class Tls(threading.local):
    def __init__(self) -> None:
        self.playwright = sync_playwright().start()


class Worker:
    def __init__(self, url, selector_wait, path) -> None:
        self.url = url
        self.selector_wait = selector_wait
        self.path = path

    tls = Tls()

    def run(self):
        try:
            # print("Launched worker in ", threading.current_thread().name)
            browser = self.tls.playwright.chromium.launch()
            page = browser.new_page()
            page.goto(self.url)
            page.wait_for_selector(self.selector_wait)
            time.sleep(1.5)
            page.screenshot(path=self.path)
            browser.close()
            time.sleep(5)
        except Exception as e:
            print(f"URL: {self.url}. Error: {e}")
            time.sleep(5)


def main():
    load_dotenv()
    # Read changesets
    root_path = os.environ["ROOT_PATH"]
    changeset_file_path = os.environ["CHANGESET_PATH"]
    n = os.environ["N"]

    changesets = pl.read_parquet(f"{root_path}/{changeset_file_path}").select(
        "changeset_id", "label", "vandalism_type", "method", "metadata"
    )

    changesets = changesets.with_columns(
        pl.col("metadata").map_elements(
            lambda x: ast.literal_eval(x), return_dtype=pl.Object
        )
    )

    # Sample positive and negative changesets equally
    positive_labels = changesets.filter(pl.col("label")).sample(n=n, seed=0)
    negative_labels = changesets.filter(~pl.col("label")).sample(n=n, seed=0)

    positive_urls = [
        (
            f"https://overpass-api.de/achavi/?changeset={id}",
            f"{root_path}/images/positive/{id}.png",
        )
        for id in positive_labels["changeset_id"]
    ]
    negative_urls = [
        (
            f"https://overpass-api.de/achavi/?changeset={id}",
            f"{root_path}/images/negative/{id}.png",
        )
        for id in negative_labels["changeset_id"]
    ]

    # Check which files to download if they don't exist
    urls = positive_urls + negative_urls
    urls = [(url, path) for url, path in urls if not os.path.exists(path)]

    selector = "#OpenLayers_Layer_Vector_113_root"

    with ThreadPoolExecutor(max_workers=10) as executor:
        for url, path in urls:
            worker = Worker(url, selector, path)
            executor.submit(worker.run)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Time taken: {t2 - t1}")
