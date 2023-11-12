import requests
import re
import os
import itertools
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
from collections import Counter
from dateutil.relativedelta import relativedelta
from datetime import datetime
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests

nltk.download("punkt")
nltk.download("stopwords")


def inference(query):
    keywords_amount = 5
    keywordList = []

    def remove_special_characters(queryText):
        pattern = r"[^a-zA-Z0-9\s]"
        cleaned_string = re.sub(pattern, "", queryText)
        return cleaned_string

    def prioritize_keywords(t, field_keywords):
        keywords_with_weights = {}

        for keyword in field_keywords:
            if keyword in freq_dist:
                keywords_with_weights[keyword] = freq_dist[keyword] * 50

        return keywords_with_weights

    
    text = remove_special_characters(query)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    freq_dist = FreqDist(filtered_tokens)

    most_common = dict(freq_dist.most_common(keywords_amount))

    field_keywords = [
            "sustainable",
            "stainless steel",
            "recyclable",
            "patent",
            "energy",
            "building",
            "infrastructure",
            "restructuring",
            "measures",
            "responsible",
            "steel",
            "efficient",
            "carbon",
            "clean",
            "waste",
            "green",
            "environmental",
            "innovation",
            "ethical",
            "materials",
            "raw",
            "industry",
            "price",
            "range",
            "investment",
            "money",
            "inflation",
            "trading",
            "news",
        ]

    keywords_with_weights = prioritize_keywords(
        remove_special_characters(query), field_keywords
    )

    combined_keywords = dict(Counter(most_common) + Counter(keywords_with_weights))

    sorted_ck = dict(
        Counter(
            dict(
                sorted(combined_keywords.items(), key=lambda x: x[1], reverse=True)
            )
        ).most_common(keywords_amount)
    )

    for keyword in sorted_ck:
        keywordList.append(keyword)


    try:
        from googlesearch import search
    except ImportError:
        print("No module named 'google' found")

    generator = search(query, tld="co.in", num=6, stop=6, pause=2)
    urls = []

    for j in generator:
        urls.append(j)
        print(j)

    index = 0
    print(urls)

    website_names = [urlparse(url).netloc[4:] for url in urls]

    class MySpider(CrawlSpider):
        name = "JunctionCrawling"
        allowed_domains = website_names
        start_urls = urls
        allowed = keywordList
        max_pages_per_website = 3
        crawled_pages_per_website = {}
        current_index = 0

        rules = (
            Rule(LinkExtractor(allow=allowed), callback="parse_item", follow=True),
        )

        def parse_item(self, response):
            website_name = urlparse(response.url).netloc[4:]

            try:
                _, _, files = next(os.walk("./" + website_name))
            except:
                self.crawled_pages_per_website[website_name] = 0
            else:
                self.crawled_pages_per_website[website_name] = len(files)
            print(self.crawled_pages_per_website[website_name])

            if (
                self.crawled_pages_per_website[website_name]
                > self.max_pages_per_website
            ):
                pass
            else:
                if all(
                    count >= self.max_pages_per_website
                    for count in self.crawled_pages_per_website.values()
                ):
                    self.crawler.engine.close_spider(
                        self, "Reached maximum pages per website"
                    )

                Path("./" + website_name).mkdir(parents=True, exist_ok=True)
                filename = "" + response.url.split("/")[-2] + ".html"
                file_path = "./" + website_name + "/" + filename
                body_content = response.xpath("//body").get()
                with open(file_path, "w") as f:
                    f.write(body_content)
                    if website_name not in self.crawled_pages_per_website:
                        self.crawled_pages_per_website[website_name] = 1
                    else:
                        self.crawled_pages_per_website[website_name] += 1

    process = CrawlerProcess(
        # settings={'CLOSESPIDER_PAGECOUNT': 100,'USER_AGENT': 'my-cool-project'}
        settings={"CLOSESPIDER_TIMEOUT": 20, "USER_AGENT": "SustainabLLM"}
    )

    process.crawl(MySpider)
    process.start()

    best_index_scores = {}
    best_website_page = {}
    best_text = ""

    for website_name in set(website_names):
        fileStrings = []
        try:
            for filename in os.listdir(
                "./" + website_name
            ):  # iterate over all files of given website
                f = os.path.join("./" + website_name, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    with open(f, "r") as htmlBody:
                        content = htmlBody.read()

                    soup = BeautifulSoup(content, "html.parser")
                    tags = soup.find_all(["p", "span"])

                    text = "".join([tag.get_text() for tag in tags])

                    fileStrings.append(text)

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(fileStrings)

            keyword_matrix = vectorizer.transform(keywordList)

            similarity_scores = cosine_similarity(
                keyword_matrix, tfidf_matrix, dense_output=True
            )

            best_match = np.mean(similarity_scores, axis=0)

            best_index = best_match.argmax()  # index for the best page within website
            best_score = best_match.max()

            best_website_page[website_name] = fileStrings[best_index]
            best_text = best_text + best_website_page[website_name]

            best_index_scores[website_name] = best_score
        except:
            print(f"{website_name} was not scraped")

    try:
        # best_text = best_website_page[max(best_index_scores, key=best_index_scores.get)]
        return best_text
    except Exception as e:
        print(e)
