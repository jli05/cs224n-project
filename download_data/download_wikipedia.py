from __future__ import print_function, unicode_literals
from BeautifulSoup import BeautifulSoup
import urllib
import urllib2
import nltk
import re
import argparse
import os
import json
import codecs
from Queue import Queue
import wikipedia

def main():
	#simple_test()
	#article_list_test()
        #w = WikipediaDownloader()
        #w.downloadArticleList()
        parser = argparse.ArgumentParser()
        parser.add_argument('page', help='wikipedia page title')
        parser.add_argument('data_path', help='output directory')
        parser.add_argument('--sleep', type=int, default=0, help='time to sleep in seconds')
        args = parser.parse_args()
        WikipediaDownloader().downloadArticle(args.page, args.data_path)

def simple_test():
	article_list = ["Chin", "Albert_Einstein", "America", "Utopia", "adsfasdfasdf"]
	for a in article_list:
		s = download_articles(a)
		if s is not None:
			print( s[0] )
			print( s[1] )

class WikipediaDownloader(object):
    def __init__(self):
        self.article_list = []

    def downloadArticleList(self, num_articles):
        seed_article = "Albert Einstein"
        self.article_list.append(seed_article)
        wikipedia.set_lang('simple')

        index = 0

        while len(self.article_list) < num_articles:
            if index >= len(self.article_list): break
            article = self.article_list[index]
            try:
                wp = wikipedia.page(article)
            except wikipedia.exceptions.PageError, e:
                index += 1
                continue
            except wikipedia.exceptions.DisambiguationError, e:
                index += 1
                continue
            links = wp.links
            self.article_list.extend(links)
            index += 1
            
        print ( self.article_list )
        with open("all.subjects", 'w') as f:
            for s in self.article_list:
                try:
                    print(s, file=f)
                except UnicodeEncodeError, e:
                    pass
            f.flush()

    def downloadArticle(self, article_title, data_path):
        output_file = os.path.join(data_path, '{:}.json'.format(article_title))
        if os.path.isfile( output_file ):
            print(article_title + "exists, skipping")
            return
        try:
            wikipedia.set_lang('simple')
            summary = wikipedia.summary(article_title, sentences=1)
            wikipedia.set_lang('en')
            text = wikipedia.summary(article_title)
        except wikipedia.exceptions.PageError, e:
            return
        summary = remove_brackets(summary)
        text = remove_brackets(text)
        if not suitable_for_training(text, summary): return
        
        output = {'title': article_title, 'full_text': text, 'summary': summary}
        with codecs.open('_{}'.format(output_file), 'w', encoding="utf-8") as outfile: json.dump(output, f, ensure_ascii=False)
        #with open(output_file, 'w') as f: json.dump(unicode(output), f, ensure_ascii=False)

# define whether or not we want to put a pair of
# sentences into our training corpus
def suitable_for_training(normal, simple):
	if len(normal) <= len(simple):
		return False
	if "|" in normal or "|" in simple: # we use these to separate sentences in our files, so we don't want them in our sentences
		return False

	# some phrases that indicate we've hit a disambiguation page... skipping these for now
	# need to make this into a list and just check the list...
	a = "For other uses"
	b = "usually refers to"
	c = "may refer to"
	d = "See List of"
	e = "Index of"
	f = "oordinates"
	g = '.'
	if a in normal or a in simple: return False
	if b in normal or b in simple: return False
	if c in normal or c in simple: return False
	if d in normal or d in simple: return False
	if e in normal or e in simple: return False
	if f in normal or f in simple: return False
	if g not in normal or g not in simple: return False
	return True

def download_articles(article_title):
	article_title = article_title.replace(" ", "_")
	title = urllib.quote(article_title)	
	opener = urllib2.build_opener()
	opener.addheaders = [('User-agent', 'Mozilla/5.0')]
	try:
		normal_sentence, _ = download_wikipedia(opener, title)
		simple_sentence, next_pages = download_wikipedia(opener, title, subwiki="simple")
		if normal_sentence is None or simple_sentence is None: return None, None
	except urllib2.HTTPError, e:
		print("No article found for " + article_title + "on one of the wikipedias... returning none")
		return None, None
	return (normal_sentence, simple_sentence), next_pages

def download_wikipedia(opener, article, subwiki='en'):
	r = opener.open("http://" + subwiki + ".wikipedia.org/wiki/" + article)
	data = r.read()
	r.close()
	soup = BeautifulSoup(data)
	text = soup.find('div', id='bodyContent').p.getText(separator=u' ')

	links = soup.findAll('a', href=True)
	possible_pages = process_links(links, subwiki)

	# we use this to pick sentences out of our text... hard to do
	# without a full natural language parser...
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	try: first_sentence = sent_detector.tokenize(text)[0]
	except IndexError, e: return None, None

	# then we remove anything between brackets...
	processed_first_sentence = remove_brackets(first_sentence) 
	processed_first_sentence = processed_first_sentence.replace("  ", " ")
	processed_first_sentence = processed_first_sentence.replace("&#160;", " ")
	return processed_first_sentence, possible_pages

def process_links(links, subwiki):
	skip_if_contains = ['wikipedia.org', 'wikimedia.org', 'Wikipedia', 'Portal', 'Privacy_policy',
			'Category', 'Special', 'Main_Page', 'Help', 'File', 'Terms_of_Use', 'Talk', 'Template',
			'#sitelinks-wikipedia']
	possible_next_pages = []
	for link in links:
		#print( link, link['href'] )
		href = link['href']
		cont = False 
		for s in skip_if_contains:
			if s in href:
				cont = True
				break
		if cont:  continue
		article_sub_url = href.split('/wiki/')
		if len(article_sub_url) > 1:
			article_title = article_sub_url[-1]
			if len(article_title) <= 1: continue
			if article_title[0] == 'Q': continue
			possible_next_pages.append(article_title)

	return possible_next_pages

def remove_brackets(text, brackets="()[]"):
	count = [0] * (len(brackets) // 2) # count open/close brackets
	saved_chars = []
	for character in text:
		for i, b in enumerate(brackets):
			if character == b: # found bracket
				kind, is_close = divmod(i, 2)
				count[kind] += (-1)**is_close # `+1`: open, `-1`: close
				if count[kind] < 0:
					count[kind] = 0
				break
		else:
			if not any(count):
				saved_chars.append(character)
	return ''.join(saved_chars)


if __name__=="__main__":
	main()


