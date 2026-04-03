
import requests
from bs4 import BeautifulSoup
import time
import os
import random

# Base configurations
BASE_URL = "https://www.dienmayxanh.com"
START_URL = "https://www.dienmayxanh.com/vao-bep/cong-thuc"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "dish_list.txt")

# Set to store unique dish names (deduplication)
found_dishes = set()

def get_soup(url):
    try:
        time.sleep(random.uniform(1, 2))  
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return BeautifulSoup(resp.content, "html.parser")
    except Exception as e:
        print(f" Error fetching {url}: {e}")
    return None

def crawl_category(category_url):
    """
    Crawls all pages of a specific category
    """
    page = 1
    while True:
        url = f"{category_url}?page={page}" if page > 1 else category_url
        print(f"   Scanning page {page}...")
        
        soup = get_soup(url)
        if not soup:
            break
            
        recipes = soup.select(".cate-cook li")
        
        
        if not recipes:
            print("  Example: No recipes found or end of pagination.")
            break
            
        new_items = 0
        for item in recipes:
            title_tag = item.select_one("strong")
            
            if not title_tag:
                 title_tag = item.select_one("h3") or item.select_one(".title")

            if title_tag:
                raw_name = title_tag.get_text(strip=True)
                lower_name = raw_name.lower()
                
                prefix_found = False
                clean_name = raw_name
                
                if lower_name.startswith("cách làm"):
                    clean_name = raw_name[8:].strip() 
                    prefix_found = True
                elif lower_name.startswith("cách nấu"):
                    clean_name = raw_name[8:].strip() 
                    prefix_found = True
                
                if prefix_found and clean_name:
                    
                    if clean_name not in found_dishes:
                        found_dishes.add(clean_name)
                        new_items += 1
        
        print(f"    + Found {new_items} new dishes.")
        
        next_btn = soup.select_one(".paging a.next") or soup.select_one(".pagging a.next")
        if not next_btn or new_items == 0:
            break
            
        page += 1
        if page > 50: 
            break

def scan_main_categories():
    print(f" Starting Discovery from {START_URL}")
    soup = get_soup(START_URL)
    if not soup:
        print(" Could not access main page.")
        return

    links = set()
    
    # Updated selector based on user provided structure: .menu-cooking.topmenu -> ul -> li -> a
    category_links = soup.select(".menu-cooking.topmenu ul li a")
    
    print(f" Found {len(category_links)} potential links in menu.")

    for a in category_links:
        href = a.get('href', '')
        
        if not href or "javascript" in href or not href.strip():
            continue
            
        if not href.startswith("/vao-bep/"):
             continue

        if "meo-vao-bep" in href:
            continue
            
        full_url = BASE_URL + href if href.startswith("/") else href
        
        if full_url.rstrip("/") == START_URL.rstrip("/"):
            continue

        if full_url not in links:
            links.add(full_url)
            
    print(f" Filtered down to {len(links)} valid categories.")
    
    for i, link in enumerate(sorted(links)):
        print(f"\n Processing Category [{i+1}/{len(links)}]: {link}")
        crawl_category(link)

def save_to_file():
    print(f"\n Saving {len(found_dishes)} dishes to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for dish in sorted(found_dishes):
            f.write(f"{dish}\n")
    print(" Done.")

if __name__ == "__main__":
    scan_main_categories()
    save_to_file()
