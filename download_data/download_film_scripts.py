from __future__ import print_function
import wikipedia
import bs4
import requests
import argparse
import re
import json
import time
import os
from data_film_scripts import downloadFilm

def download_film_scripts(titles_file_path, output_path, sleep=0):
    with open(titles_file_path, 'r') as titles_file:
        for line in titles_file:
            m = re.match(r'"(.*)"', line.strip())
            if not m:
                continue
            try:
                downloadFilm(m.group(1), output_path)
                time.sleep(sleep)
            except Exception as e:
                print(e)
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_file',
                        help='file with all the film titles to download')
    parser.add_argument('output_path',
                        help='directory for output')
    parser.add_argument('--sleep', type=int,
                        default=0,
                        help='time to sleep after download of each title (in seconds)')

    args = parser.parse_args()
    download_film_scripts(args.titles_file, args.output_path, args.sleep)

if __name__ == '__main__':
    main()
    
