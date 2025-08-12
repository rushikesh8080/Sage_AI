import requests
from bs4 import BeautifulSoup
import time
import json

BASE_URL = "https://plato.stanford.edu"
CHRONOLOGY_URL = f"{BASE_URL}/published.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def get_philosopher_links():
    response = requests.get(CHRONOLOGY_URL, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch the contents page! Status Code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    entries = soup.find_all("li")
    
    philosopher_links = []
    for entry in entries:
        link_tag = entry.find("a")
        if link_tag and "entries" in link_tag["href"]:
            title = link_tag.text.strip()
            link = f"{BASE_URL}{link_tag['href']}" if link_tag["href"].startswith("/entries") else link_tag["href"]
            philosopher_links.append((title, link))
    return philosopher_links

def scrape_philosopher(url):
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Skipping {url} - Status Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    title_tag = soup.find("h1")
    title = title_tag.text.strip() if title_tag else "Unknown Title"
    
    sections = {}
    current_section = "Introduction"
    sections[current_section] = []
    
    # Unwanted sections
    unwanted_sections = ["Bibliography", "Academic Tools", "Other Internet Resources", "Related Entries", 
                         "Browse", "About", "Support SEP", "Mirror Sites"]
    
    for tag in soup.find_all(['h2', 'h3', 'p']):
        if tag.name in ['h2', 'h3']:
            section_title = tag.text.strip()
            if section_title in unwanted_sections:
                break
            current_section = section_title
            sections[current_section] = []
        elif tag.name == 'p' and current_section:
            sections[current_section].append(tag.text.strip())
    
    return {
        "title": title,
        "url": url,
        "sections": sections
    }

def main():
    philosopher_links = get_philosopher_links()
    if not philosopher_links:
        print("No philosopher links found!")
        return
    
    philosopher_data = {}
    for title, url in philosopher_links[:1]:  
        print(f"Scraping: {title} -> {url}")
        data = scrape_philosopher(url)
        if data:
            philosopher_data[url] = data
        time.sleep(1)
    
    with open("sample.json", "w", encoding="utf-8") as f:
        json.dump(philosopher_data, f, ensure_ascii=False, indent=4)
    
    print("Scraping complete! Data saved to 'philosophers.json'.")

if __name__ == "__main__":
    main()