#!/usr/bin/env python3
"""
Scrape STE Dictionary from https://ste.valaratomics.com/dictionary/
Store in SQLite and export to CSV.
"""

import sqlite3
import csv
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://ste.valaratomics.com/dictionary/"
TOTAL_PAGES = 299
DB_PATH = Path(__file__).parent / "ste_dictionary.db"
CSV_PATH = Path(__file__).parent / "ste_dictionary.csv"


def create_database():
    """Create SQLite database with dictionary table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            part_of_speech TEXT,
            is_ste INTEGER,
            approved_meaning TEXT,
            ste_example TEXT,
            non_ste_example TEXT
        )
    """)

    conn.commit()
    return conn


def parse_word_pos(text):
    """Parse word and part of speech from text like 'abandon (verb)'."""
    text = text.strip()
    if '(' in text and text.endswith(')'):
        parts = text.rsplit('(', 1)
        word = parts[0].strip()
        pos = parts[1].rstrip(')').strip()
        return word, pos
    return text, None


def scrape_page(page_num, session):
    """Scrape a single page and return list of entries."""
    url = f"{BASE_URL}?page={page_num}"

    for attempt in range(4):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 3:
                wait_time = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/4 after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after 4 attempts: {e}")
                return []

    soup = BeautifulSoup(response.text, 'html.parser')
    entries = []

    # Find the table - try different approaches
    table = soup.find('table')
    if not table:
        print(f"  No table found on page {page_num}")
        return []

    rows = table.find_all('tr')

    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 5:
            # Skip header row
            first_cell = cells[0].get_text(strip=True)
            if first_cell.lower() == 'word (part of speech)':
                continue

            word_pos = cells[0].get_text(strip=True)
            word, pos = parse_word_pos(word_pos)

            ste_indicator = cells[1].get_text(strip=True)
            is_ste = 1 if '✔' in ste_indicator or 'check' in ste_indicator.lower() else 0

            approved_meaning = cells[2].get_text(separator='\n', strip=True)
            ste_example = cells[3].get_text(strip=True)
            non_ste_example = cells[4].get_text(strip=True)

            if word:  # Only add if we have a word
                entries.append({
                    'word': word,
                    'part_of_speech': pos,
                    'is_ste': is_ste,
                    'approved_meaning': approved_meaning,
                    'ste_example': ste_example,
                    'non_ste_example': non_ste_example
                })

    return entries


def insert_entries(conn, entries):
    """Insert entries into database."""
    cursor = conn.cursor()

    for entry in entries:
        cursor.execute("""
            INSERT INTO dictionary (word, part_of_speech, is_ste, approved_meaning, ste_example, non_ste_example)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry['word'],
            entry['part_of_speech'],
            entry['is_ste'],
            entry['approved_meaning'],
            entry['ste_example'],
            entry['non_ste_example']
        ))

    conn.commit()


def export_to_csv(conn):
    """Export database to CSV file."""
    cursor = conn.cursor()
    cursor.execute("SELECT word, part_of_speech, is_ste, approved_meaning, ste_example, non_ste_example FROM dictionary ORDER BY word")

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'part_of_speech', 'is_ste', 'approved_meaning', 'ste_example', 'non_ste_example'])
        writer.writerows(cursor.fetchall())

    print(f"Exported to {CSV_PATH}")


def main():
    print("Creating database...")
    conn = create_database()

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; DictionaryScraper/1.0)'
    })

    total_entries = 0

    print(f"Scraping {TOTAL_PAGES} pages...")
    for page in range(1, TOTAL_PAGES + 1):
        print(f"Page {page}/{TOTAL_PAGES}...", end=" ")
        entries = scrape_page(page, session)

        if entries:
            insert_entries(conn, entries)
            total_entries += len(entries)
            print(f"{len(entries)} entries (total: {total_entries})")
        else:
            print("no entries")

        # Be polite to the server
        time.sleep(0.5)

    print(f"\nTotal entries scraped: {total_entries}")

    print("Exporting to CSV...")
    export_to_csv(conn)

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
