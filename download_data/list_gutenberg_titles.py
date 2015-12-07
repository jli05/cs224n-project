from __future__ import (division, absolute_import,
                        print_function, unicode_literals)

import requests
from bs4 import BeautifulSoup
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from data_scripts import downloadSummary
import time
import argparse

def list_most_popular_titles(n, start_index=1, sleep=0):
    INDEX_PAGE = 'https://www.gutenberg.org/ebooks/search/?sort_order=downloads&start_index={:}'
    
    n_batch = int(n / 25)
    for i in range(n_batch):
        index_page = INDEX_PAGE.format(i * 25 + start_index)
        r = requests.get(index_page)
        soup = BeautifulSoup(r.text, 'lxml')
        
        links = soup.findAll('li', {'class': 'booklink'})
        book_titles = [link.a.find('span', {'class': 'title'}).text.strip() for link in links]
        gutenberg_ids = [int(link.a['href'].split('/')[2]) for link in links]

        for title, gutenberg_id in zip(book_titles, gutenberg_ids):
            print('"{:}",{:}'.format(title, gutenberg_id))

        time.sleep(sleep)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int,
                        help='number of titles to list')
    parser.add_argument('--start-index', type=int,
                        default=1,
                        help='start index')
    parser.add_argument('--sleep', type=int,
                        default=0,
                        help='sleep between page load (in seconds)')
    args = parser.parse_args()

    list_most_popular_titles(args.n, args.start_index, args.sleep)

if __name__ == '__main__':
    main()
        
    

    

    
