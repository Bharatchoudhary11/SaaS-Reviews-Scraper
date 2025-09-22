#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

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
	reviews: List[Dict] = []

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
	reviews: List[Dict] = []

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


def build_g2_urls(company_slug: str, max_pages: int = 20) -> List[str]:
	# Common G2 products path: https://www.g2.com/products/{slug}/reviews?page=1
	return [f"https://www.g2.com/products/{company_slug}/reviews?page={i}" for i in range(1, max_pages + 1)]


def build_capterra_urls(company_slug: str, max_pages: int = 20) -> List[str]:
	# Capterra URLs vary. A generic fallback that sometimes works:
	# https://www.capterra.com/reviews/{slug}?page=1
	# We also try alternate pattern used on localized pages.
	candidates = []
	for i in range(1, max_pages + 1):
		candidates.append(f"https://www.capterra.com/reviews/{company_slug}?page={i}")
		candidates.append(f"https://www.capterra.com/{company_slug}/reviews/?page={i}")
	return candidates


def scrape_reviews(company: str, source: str, start: Optional[datetime], end: Optional[datetime], max_pages: int = 20) -> List[Dict]:
	session = requests.Session()
	company_slug = normalize_company_to_slug(company)

	if source.lower() == "g2":
		urls = build_g2_urls(company_slug, max_pages=max_pages)
		extractor = extract_reviews_g2
	elif source.lower() == "capterra":
		urls = build_capterra_urls(company_slug, max_pages=max_pages)
		extractor = extract_reviews_capterra
	else:
		raise ValueError("source must be either 'G2' or 'Capterra'")

	all_reviews: List[Dict] = []

	for url in tqdm(urls, desc=f"Scraping {source} pages", unit="page"):
		html = fetch_html(url, session=session)
		if not html:
			continue
		page_reviews = extractor(html)
		if not page_reviews:
			# Heuristic: continue; sites may still have reviews on later pages
			continue
		all_reviews.extend(page_reviews)
		# Optional: be respectful
		time.sleep(0.75)

	# Deduplicate by (title, review, date)
	unique: Dict[str, Dict] = {}
	for r in all_reviews:
		key = json.dumps([(r.get("title") or "").strip(), (r.get("review") or "").strip(), (r.get("date") or "").strip()])
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

	return filtered


def main():
	parser = argparse.ArgumentParser(description="Scrape product reviews from G2 or Capterra and output JSON.")
	parser.add_argument("--company", required=True, help="Company/Product name to scrape reviews for")
	parser.add_argument("--start", required=True, help="Start date (e.g., 2021-01-01)")
	parser.add_argument("--end", required=True, help="End date (e.g., 2021-12-31)")
	parser.add_argument("--source", required=True, choices=["G2", "Capterra", "g2", "capterra"], help="Review source")
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
