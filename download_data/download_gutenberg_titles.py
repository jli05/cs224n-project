from __future__ import (division, absolute_import,
                        print_function, unicode_literals)

import requests
from bs4 import BeautifulSoup
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from data_film_scripts import downloadSummary
from data_gutenberg import download_book
import time
import json
import argparse
import os
import re

def download_books(titles_file_path, output_path, sleep=0):
    with open(titles_file_path, 'r') as titles_file:
        for line in titles_file:
            m = re.match(r'"(.*)",(\d+)', line.strip())
            if not m:
                continue
            try:
                download_book(m.group(1), int(m.group(2)), output_path, sleep)
            except Exception as e:
                print(e)
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_file',
                        help='the file containing the Gutenberg titles for download')
    parser.add_argument('output_path',
                        help='output directory')
    parser.add_argument('--sleep', type=int,
                        default=0,
                        help='time to sleep (in seconds)')
    args = parser.parse_args()
    
    download_books(args.titles_file, args.output_path, args.sleep)

if __name__ == '__main__':
    main()

