#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Set
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm


DEFAULT_HEADERS = {
        "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
}


def parse_iso_date(value: str) -> Optional[datetime]:
        if not value:
                return None
        try:
                return dateparser.parse(value)
        except Exception:
                return None


def in_range(dt: Optional[datetime], start: Optional[datetime], end: Optional[datetime]) -> bool:
        if dt is None:
                return False
        if start and dt < start:
                return False
        if end and dt > end:
                return False
        return True


def normalize_company_to_slug(company: str) -> str:
        slug = company.strip().lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug


def _normalize_value(value: object) -> Optional[str]:
        if value is None:
                return None
        if isinstance(value, str):
                candidate = value.strip()
                return candidate or None
        if isinstance(value, (list, tuple)):
                for item in value:
                        candidate = _normalize_value(item)
                        if candidate:
                                return candidate
                return None
        if isinstance(value, dict):
                # Attempt common textual keys
                for key in ("name", "text", "title", "description"):
                        if key in value:
                                candidate = _normalize_value(value[key])
                                if candidate:
                                        return candidate
                return None
        candidate = str(value).strip()
        return candidate or None


def _first_non_empty(*values: object) -> Optional[str]:
        for value in values:
                candidate = _normalize_value(value)
                if candidate:
                        return candidate
        return None


def _iter_jsonld_reviews(data: object) -> Iterable[Dict]:
        if isinstance(data, list):
                for item in data:
                        yield from _iter_jsonld_reviews(item)
        elif isinstance(data, dict):
                type_value = data.get("@type")
                if isinstance(type_value, str) and type_value.lower() == "review":
                        yield data
                elif isinstance(type_value, list) and any(
                        isinstance(t, str) and t.lower() == "review" for t in type_value
                ):
                        yield data

                for value in data.values():
                        if isinstance(value, (dict, list)):
                                yield from _iter_jsonld_reviews(value)


def extract_reviews_from_json_ld(soup: BeautifulSoup, source: str) -> List[Dict]:
        reviews: List[Dict] = []
        for script in soup.find_all("script", {"type": "application/ld+json"}):
                if not script.string:
                        continue
                try:
                        data = json.loads(script.string)
                except json.JSONDecodeError:
                        continue

                for review_node in _iter_jsonld_reviews(data):
                        rating_value = review_node.get("reviewRating") or review_node.get("aggregateRating")
                        if isinstance(rating_value, dict):
                                rating_value = rating_value.get("ratingValue") or rating_value.get("value")

                        author = review_node.get("author")
                        if isinstance(author, dict):
                                reviewer_name = _first_non_empty(author.get("name"), author.get("alternateName"))
                        else:
                                reviewer_name = _first_non_empty(author)

                        reviews.append(
                                {
                                        "title": _first_non_empty(review_node.get("name"), review_node.get("headline")),
                                        "review": _first_non_empty(
                                                review_node.get("reviewBody"), review_node.get("description")
                                        ),
                                        "date": _first_non_empty(
                                                review_node.get("datePublished"),
                                                review_node.get("dateCreated"),
                                                review_node.get("date"),
                                        ),
                                        "rating": _first_non_empty(review_node.get("ratingValue"), rating_value),
                                        "reviewer": reviewer_name,
                                        "source": source,
                                }
                        )

        return [review for review in reviews if review.get("title") or review.get("review") or review.get("date")]


def fetch_html(url: str, session: requests.Session, retries: int = 3, backoff: float = 1.5) -> Optional[str]:
        for attempt in range(retries):
                try:
                        resp = session.get(url, headers=DEFAULT_HEADERS, timeout=20)
                        if resp.status_code == 200:
                                return resp.text
                        # Retry on 429/5xx
                        if resp.status_code in (429, 500, 502, 503, 504):
                                time.sleep(backoff ** attempt)
                                continue
                        return None
                except requests.RequestException:
                        time.sleep(backoff ** attempt)
        return None


def extract_reviews_g2(html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        reviews: List[Dict] = extract_reviews_from_json_ld(soup, "G2")

        # Prefer schema.org-style extraction
        review_containers = soup.select('[itemtype="http://schema.org/Review"], [itemtype="https://schema.org/Review"]')
        if not review_containers:
                # fallback to common class names
                review_containers = soup.select(".review, .paper.review")

        for rc in review_containers:
                title = None
                title_el = rc.select_one('[itemprop="name"]') or rc.select_one(".review-title, h3, h2")
                if title_el:
                        title = title_el.get_text(strip=True) or None

                body = None
                body_el = rc.select_one('[itemprop="reviewBody"]') or rc.select_one(".review-body, .content, p")
                if body_el:
                        body = body_el.get_text(" ", strip=True) or None

                date_val = None
                date_el = rc.select_one('[itemprop="datePublished"]') or rc.find("time")
                if date_el:
                        date_val = date_el.get("datetime") or date_el.get_text(strip=True)

                rating_val = None
                rating_el = rc.select_one('[itemprop="ratingValue"]') or rc.select_one(".rating, .stars")
                if rating_el:
                        rating_val = rating_el.get("content") or rating_el.get_text(strip=True)

                reviewer = None
                reviewer_el = rc.select_one('[itemprop="author"] [itemprop="name"]') or rc.select_one(".user, .author, .reviewer")
                if reviewer_el:
                        reviewer = reviewer_el.get_text(strip=True)

                if not (title or body or date_val):
                        continue

                reviews.append({
                        "title": title,
                        "review": body,
                        "date": date_val,
                        "rating": rating_val,
                        "reviewer": reviewer,
                        "source": "G2",
                })

        return reviews


def extract_reviews_capterra(html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        reviews: List[Dict] = extract_reviews_from_json_ld(soup, "Capterra")

        review_containers = soup.select('[itemtype="http://schema.org/Review"], [itemtype="https://schema.org/Review"], .review-card, .review')

        for rc in review_containers:
                title = None
                title_el = rc.select_one('[itemprop="name"]') or rc.select_one(".review-title, h3, h2")
                if title_el:
                        title = title_el.get_text(strip=True) or None

                body = None
                body_el = rc.select_one('[itemprop="reviewBody"]') or rc.select_one(".review-body, .content, p")
                if body_el:
                        body = body_el.get_text(" ", strip=True) or None

                date_val = None
                date_el = rc.select_one('[itemprop="datePublished"]') or rc.find("time")
                if date_el:
                        date_val = date_el.get("datetime") or date_el.get_text(strip=True)

                rating_val = None
                rating_el = rc.select_one('[itemprop="ratingValue"]') or rc.select_one(".rating, .stars")
                if rating_el:
                        rating_val = rating_el.get("content") or rating_el.get_text(strip=True)

                reviewer = None
                reviewer_el = rc.select_one('[itemprop="author"] [itemprop="name"]') or rc.select_one(".user, .author, .reviewer")
                if reviewer_el:
                        reviewer = reviewer_el.get_text(strip=True)

                if not (title or body or date_val):
                        continue

                reviews.append({
                        "title": title,
                        "review": body,
                        "date": date_val,
                        "rating": rating_val,
                        "reviewer": reviewer,
                        "source": "Capterra",
                })

        return reviews


def extract_reviews_trustradius(html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        reviews: List[Dict] = extract_reviews_from_json_ld(soup, "TrustRadius")

        review_containers = soup.select(
                '[itemtype="http://schema.org/Review"], '
                '[itemtype="https://schema.org/Review"], '
                'article.review, article[class*="Review"], '
                'div.review, div.review-card, div[class*="review-card"], '
                'section.review, li.review'
        )

        for rc in review_containers:
                title = None
                title_el = rc.select_one('[itemprop="name"]') or rc.select_one(".review-title, h3, h2")
                if title_el:
                        title = title_el.get_text(strip=True) or None

                body = None
                body_el = rc.select_one('[itemprop="reviewBody"]') or rc.select_one(".review-body, .content, p")
                if body_el:
                        body = body_el.get_text(" ", strip=True) or None

                date_val = None
                date_el = rc.select_one('[itemprop="datePublished"]') or rc.find("time")
                if date_el:
                        date_val = date_el.get("datetime") or date_el.get_text(strip=True)

                rating_val = None
                rating_el = rc.select_one('[itemprop="ratingValue"]') or rc.select_one(".rating, .stars")
                if rating_el:
                        rating_val = rating_el.get("content") or rating_el.get_text(strip=True)

                reviewer = None
                reviewer_el = rc.select_one('[itemprop="author"] [itemprop="name"]') or rc.select_one(
                        ".user, .author, .reviewer"
                )
                if reviewer_el:
                        reviewer = reviewer_el.get_text(strip=True)

                if not (title or body or date_val):
                        continue

                reviews.append(
                        {
                                "title": title,
                                "review": body,
                                "date": date_val,
                                "rating": rating_val,
                                "reviewer": reviewer,
                                "source": "TrustRadius",
                        }
                )

        return reviews


def build_g2_urls(company_slug: str, max_pages: int = 20) -> List[str]:
        # Common G2 products path: https://www.g2.com/products/{slug}/reviews?page=1
        return [f"https://www.g2.com/products/{company_slug}/reviews?page={i}" for i in range(1, max_pages + 1)]


def _collect_slugs_from_json(node: object, results: Set[str]) -> None:
        if isinstance(node, dict):
                slug_candidate = node.get("slug")
                if isinstance(slug_candidate, str):
                        slug_candidate = slug_candidate.strip()
                        if slug_candidate:
                                results.add(slug_candidate)

                path_candidate = node.get("path") or node.get("url") or node.get("href")
                if isinstance(path_candidate, str):
                        match = re.search(r"/products/([^/]+)/reviews", path_candidate)
                        if match:
                                results.add(match.group(1).strip())

                for value in node.values():
                        _collect_slugs_from_json(value, results)
        elif isinstance(node, list):
                for item in node:
                        _collect_slugs_from_json(item, results)
        elif isinstance(node, str):
                match = re.search(r"/products/([^/]+)/reviews", node)
                if match:
                        results.add(match.group(1).strip())


def _score_slug_candidate(candidate: str, normalized_query: str) -> float:
        if not candidate:
                return -math.inf

        candidate_norm = normalize_company_to_slug(candidate)
        score = 0.0

        if candidate_norm == normalized_query:
                score += 3.0
        elif normalized_query in candidate_norm:
                score += 2.0

        # Simple similarity heuristic to prioritize close matches.
        overlap = len(set(candidate_norm.split("-")) & set(normalized_query.split("-")))
        score += overlap * 0.5

        # Slight preference for shorter slugs when everything else ties.
        score -= len(candidate_norm) * 0.01
        return score


def find_g2_slug(company: str, session: requests.Session) -> Optional[str]:
        query = company.strip()
        if not query:
                return None

        normalized_query = normalize_company_to_slug(query)
        search_templates = (
                "https://www.g2.com/search?query={query}",
                "https://www.g2.com/search/products?query={query}",
        )

        best_slug: Optional[str] = None
        best_score = -math.inf

        for template in search_templates:
                search_url = template.format(query=quote_plus(query))
                html = fetch_html(search_url, session=session)
                if not html:
                        continue

                soup = BeautifulSoup(html, "html.parser")

                candidates: Set[str] = set()

                # G2 embeds useful metadata about products in data attributes.
                for attr in ("data-product-slug", "data-track-product-slug", "data-slug"):
                        for node in soup.select(f"[{attr}]"):
                                slug = _normalize_value(node.get(attr))
                                if slug:
                                        candidates.add(slug)

                for link in soup.select("a[href]"):
                        href = link.get("href") or ""
                        match = re.search(r"/products/([^/]+)/reviews", href)
                        if match:
                                slug = match.group(1).strip()
                                if slug:
                                        candidates.add(slug)

                script_node = soup.find("script", {"id": "__NEXT_DATA__"})
                if script_node and script_node.string:
                        try:
                                data = json.loads(script_node.string)
                        except json.JSONDecodeError:
                                data = None

                        if data is not None:
                                _collect_slugs_from_json(data, candidates)

                for candidate in candidates:
                        score = _score_slug_candidate(candidate, normalized_query)
                        if score > best_score:
                                best_slug = candidate
                                best_score = score

        return best_slug


def build_capterra_urls(company_slug: str, max_pages: int = 20) -> List[str]:
        # Capterra URLs vary. A generic fallback that sometimes works:
        # https://www.capterra.com/reviews/{slug}?page=1
        # We also try alternate pattern used on localized pages.
        candidates = []
        for i in range(1, max_pages + 1):
                candidates.append(f"https://www.capterra.com/reviews/{company_slug}?page={i}")
                candidates.append(f"https://www.capterra.com/{company_slug}/reviews/?page={i}")
        return candidates


def build_trustradius_urls(company_slug: str, max_pages: int = 20) -> List[str]:
        base = f"https://www.trustradius.com/products/{company_slug}/reviews"
        return [f"{base}?page={i}" if i > 1 else base for i in range(1, max_pages + 1)]


UrlBuilder = Callable[[str, int], List[str]]
SourceExtractor = Callable[[str], List[Dict]]
SlugFinder = Callable[[str, requests.Session], Optional[str]]


@dataclass(frozen=True)
class SourceConfig:
        name: str
        url_builder: UrlBuilder
        extractor: SourceExtractor
        slug_finder: Optional[SlugFinder] = None


SOURCE_REGISTRY: Dict[str, SourceConfig] = {
        "g2": SourceConfig(
                name="G2",
                url_builder=build_g2_urls,
                extractor=extract_reviews_g2,
                slug_finder=find_g2_slug,
        ),
        "capterra": SourceConfig(
                name="Capterra", url_builder=build_capterra_urls, extractor=extract_reviews_capterra
        ),
        "trustradius": SourceConfig(
                name="TrustRadius", url_builder=build_trustradius_urls, extractor=extract_reviews_trustradius
        ),
}


def scrape_reviews(
        company: str,
        source: str,
        start: Optional[datetime],
        end: Optional[datetime],
        max_pages: int = 20,
) -> List[Dict]:
        session = requests.Session()
        company_slug = normalize_company_to_slug(company)

        source_key = source.lower()
        config = SOURCE_REGISTRY.get(source_key)
        if not config:
                valid_names = ", ".join(sorted(cfg.name for cfg in SOURCE_REGISTRY.values()))
                raise ValueError(f"source must be one of: {valid_names} (case-insensitive).")

        attempted_slugs: List[str] = []
        slug_queue: List[str] = [company_slug]
        seen_slugs = {company_slug}

        def attempt_scrape_for_slug(candidate_slug: str) -> Optional[List[Dict]]:
                urls = config.url_builder(candidate_slug, max_pages=max_pages)
                extractor = config.extractor

                all_reviews: List[Dict] = []
                pages_fetched = 0
                raw_reviews_extracted = 0

                for url in tqdm(urls, desc=f"Scraping {config.name} pages", unit="page"):
                        html = fetch_html(url, session=session)
                        if not html:
                                continue
                        pages_fetched += 1
                        page_reviews = extractor(html)
                        raw_reviews_extracted += len(page_reviews)
                        if not page_reviews:
                                # Heuristic: continue; sites may still have reviews on later pages
                                continue
                        all_reviews.extend(page_reviews)
                        # Optional: be respectful
                        time.sleep(0.75)

                if pages_fetched == 0:
                        return None

                # Deduplicate by (title, review, date)
                unique: Dict[str, Dict] = {}
                for r in all_reviews:
                        key = json.dumps(
                                [
                                        (r.get("title") or "").strip(),
                                        (r.get("review") or "").strip(),
                                        (r.get("date") or "").strip(),
                                ]
                        )
                        if key not in unique:
                                unique[key] = r

                # Normalize and filter by date range
                filtered: List[Dict] = []
                for r in unique.values():
                        dt = parse_iso_date(r.get("date") or "")
                        if not in_range(dt, start, end):
                                continue
                        # Set a stable ISO format for date if we could parse it
                        if dt:
                                r["date"] = dt.strftime("%Y-%m-%d")
                        filtered.append(r)

                if raw_reviews_extracted == 0:
                        print(
                                f"Warning: {config.name} pages were fetched but no reviews could be parsed. "
                                "The site layout may have changed.",
                                file=sys.stderr,
                        )
                elif not filtered and (start or end):
                        print(
                                f"Note: {raw_reviews_extracted} reviews were extracted from {config.name}, "
                                "but none matched the requested date range.",
                                file=sys.stderr,
                        )

                return filtered

        while slug_queue:
                slug = slug_queue.pop(0)
                attempted_slugs.append(slug)
                result = attempt_scrape_for_slug(slug)
                if result is not None:
                        return result

                if slug == company_slug and config.slug_finder:
                        try:
                                alternate_slug = config.slug_finder(company, session=session)
                        except Exception:
                                alternate_slug = None
                        if alternate_slug and alternate_slug not in seen_slugs:
                                slug_queue.append(alternate_slug)
                                seen_slugs.add(alternate_slug)

        attempted_display = ", ".join(attempted_slugs)
        raise RuntimeError(
                "Unable to load any review pages. "
                "Verify the company slug exists on the selected source or try again later. "
                f"Attempted slug(s): {attempted_display or company_slug}."
        )


def main():
        parser = argparse.ArgumentParser(
                description="Scrape product reviews from G2, Capterra, or TrustRadius and output JSON."
        )
        parser.add_argument("--company", required=True, help="Company/Product name to scrape reviews for")
        parser.add_argument("--start", required=True, help="Start date (e.g., 2021-01-01)")
        parser.add_argument("--end", required=True, help="End date (e.g., 2021-12-31)")
        source_choices = sorted(
                set(SOURCE_REGISTRY.keys())
                | {config.name for config in SOURCE_REGISTRY.values()}
        )
        parser.add_argument("--source", required=True, choices=source_choices, help="Review source")
        parser.add_argument("--output", default=None, help="Output JSON file path (optional)")
        parser.add_argument("--pages", type=int, default=20, help="Maximum pages to crawl (default: 20)")

        args = parser.parse_args()

        try:
                start_dt = parse_iso_date(args.start)
                end_dt = parse_iso_date(args.end)
                if not start_dt or not end_dt:
                        raise ValueError("Invalid start or end date format. Use YYYY-MM-DD or a recognizable date string.")
                if end_dt < start_dt:
                        raise ValueError("End date must be on or after start date.")
        except Exception as e:
                print(f"Error parsing dates: {e}", file=sys.stderr)
                sys.exit(1)

        try:
                reviews = scrape_reviews(
                        company=args.company,
                        source=args.source,
                        start=start_dt,
                        end=end_dt,
                        max_pages=args.pages,
                )
        except Exception as e:
                print(f"Failed to scrape: {e}", file=sys.stderr)
                sys.exit(2)

        output_path = args.output
        if not output_path:
                safe_company = normalize_company_to_slug(args.company)
                safe_source = args.source.lower()
                output_filename = f"reviews_{safe_source}_{safe_company}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.json"
                output_path = os.path.join(os.getcwd(), output_filename)

        try:
                with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(reviews, f, ensure_ascii=False, indent=2)
                print(f"Wrote {len(reviews)} reviews to {output_path}")
        except Exception as e:
                print(f"Failed to write output: {e}", file=sys.stderr)
                sys.exit(3)


if __name__ == "__main__":
        main()
