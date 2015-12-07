from __future__ import print_function
import wikipedia
import bs4
import requests
import argparse
import re
import json
import time
import os

def findWikipediaPage(film_title):
    possibles = (wikipedia.search(film_title)
                 + wikipedia.search(film_title + ' (film)'))
    for m in possibles:
        a = re.match(r'(.*) \((\d+ )?film\)', m.lower())
        if (a and
            (a.group(1).lower() in film_title.lower()
             or film_title.lower() in a.group(1).lower())):
            return m
    try:
        if (possibles[0].lower() in film_title.lower()
            or film_title.lower() in possibles[0].lower()):
            return possibles[0]
        else:
            return None
    except Exception as e: return None

def downloadSummary(title):
    if title is None:
        return None
    wikipedia_page = wikipedia.page(title)
    content = wikipedia_page.content

    headers = re.findall(r'\n== ([\w\s]+) ==\n', content)
    for i in range(len(headers)):
        if ('plot' in headers[i].lower()
            or 'synopsis' in headers[i].lower()
            or 'summary' in headers[i].lower()):
            p1 = content.find('\n== {:} ==\n'.format(headers[i]))
            if i < len(headers) - 1:
                p2 = content.find('\n== {:} ==\n'.format(headers[i + 1]))
                summary_ = content[(p1 + 8 + len(headers[i])):p2]
            else:
                summary_ = content[(p1 + 8 + len(headers[i])):]
            # strip the subsection titles in summary_
            return re.sub(r'\n==+ .* ==+\n', '', summary_)

    return None

def downloadScript(film_title):
    r = requests.get('http://www.imsdb.com/scripts/' + film_title.title().replace(" ", "-") + '.html')
    soup = bs4.BeautifulSoup(r.text, 'lxml')
    try:
        return soup.find('td', {'class': 'scrtext'}).text
    except Exception as e:
        return None

def downloadFilm(film_title, output_path):
    wikipedia_page_title = findWikipediaPage(film_title)
    print('downloading {:}'.format(film_title))
    print('Wikipedia page {:}'.format(wikipedia_page_title))
    
    script = downloadScript(film_title)
    summary = downloadSummary(wikipedia_page_title)

    if script is None or summary is None:
        print('Error occurred. Skipping {:}'.format(film_title))
        return
        
    script = script.strip()
    summary = summary.strip()
    output_json = {'title': film_title,
                   'full_text': script,
                   'summary': summary}
        
    filename = film_title.replace(' ', '_') + '.json'
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w') as f:
        json.dump(output_json, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('title',
                        help='film title for download')
    parser.add_argument('output_path',
                        help='path for output')
    args = parser.parse_args()
    
    downloadFilm(args.title, args.output_path)

if __name__ == "__main__":
    main()
