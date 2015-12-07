from __future__ import (division, absolute_import,
                        print_function, unicode_literals)

import requests
from bs4 import BeautifulSoup
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from data_film_scripts import downloadSummary
import time
import json
import argparse
import os

def download_book(title, gutenberg_id, data_path, sleep=0):
    print('downloading {:}'.format(title))

    full_text = strip_headers(load_etext(gutenberg_id)).strip()
    summary = downloadSummary(title)

    if full_text is None:
        print('Full text is None. Skipping {:}'.format(title))
        return
    if summary is None:
        print('Summary is None. Skipping {:}'.format(title))
        return

    output_data = {'title': title,
                   'full_text': full_text,
                   'summary': summary}
        
    output_file = os.path.join(data_path,
                               '{:}.json'.format(gutenberg_id))
    with open(output_file, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False)

    time.sleep(sleep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('title',
                        help='book title')
    parser.add_argument('gutenberg_id', type=int,
                        help='Gutenberg book id')
    parser.add_argument('data_path',
                        help='output directory')
    parser.add_argument('--sleep', type=int,
                        default=0,
                        help='time to sleep (in seconds)')
    args = parser.parse_args()

    download_book(args.title, args.gutenberg_id, args.data_path, args.sleep)

if __name__ == '__main__':
    main()
    



