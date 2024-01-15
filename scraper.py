import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
from datetime import datetime
import sqlite3
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SQLite setup
conn = sqlite3.connect('scraped_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS web_pages
             (url TEXT PRIMARY KEY, content TEXT, md5 TEXT, timestamp TEXT)''')
conn.commit()

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc.endswith("docs.sbab.se") and url.endswith('.html')

def get_all_urls(url, found_urls=set()):
    try:
        response = requests.get(url, verify=False)
        print("Fetching URLs from:", url)  # Added to show progress
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return found_urls

    soup = BeautifulSoup(response.content, "html.parser")
    for link in soup.find_all("a", href=True):
        absolute_link = urljoin(url, link["href"])
        if is_valid_url(absolute_link) and absolute_link not in found_urls:
            found_urls.add(absolute_link)
            print(absolute_link)  # Print each URL found
            get_all_urls(absolute_link, found_urls)
    return found_urls

def fetch_url_content(url):
    try:
        response = requests.get(url, verify=False)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text

def calculate_md5(content):
    return hashlib.md5(content.encode()).hexdigest()

def save_content_to_file(url, content, folder="pages"):
    # Extract the filename from the URL
    file_name = url.split("/")[-1]

    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)

    # Construct the full path for the file
    file_path = os.path.join(folder, file_name)

    # Save the HTML content to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Saved HTML content to {file_path}")
    
def scrape_store_and_hash(urls):
    for url in urls:
        html_content = fetch_url_content(url)
        if html_content:
            text = extract_text_from_html(html_content)
            save_content_to_file(url, text)

            new_md5_hash = calculate_md5(text)

            # Check if URL already exists in the database
            c.execute("SELECT md5 FROM web_pages WHERE url=?", (url,))
            result = c.fetchone()

            if result is None or result[0] != new_md5_hash:
                timestamp = datetime.now().isoformat()
                c.execute('INSERT OR REPLACE INTO web_pages (url, content, md5, timestamp) VALUES (?, ?, ?, ?)',
                          (url, text, new_md5_hash, timestamp))
                print(f"Updated DB for: {url}")
            else:
                print(f"No change for: {url}")
            conn.commit()

if __name__ == "__main__":
    start_url = "https://docs.sbab.se/"
    found_urls = get_all_urls(start_url)
    scrape_store_and_hash(found_urls)
    conn.close()
