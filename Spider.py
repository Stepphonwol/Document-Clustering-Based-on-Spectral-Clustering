import requests
from bs4 import BeautifulSoup
import os


class Spider:

    def __init__(self):
        self.index = 0

    def get_html_1(self, url):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                          ' Chrome/66.0.3359.181 Safari/537.36',
            'host': 'www.douban.com'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = 'utf-8'
        return r.text

    def get_html_2(self, url):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                          ' Chrome/66.0.3359.181 Safari/537.36',
            'host': 'movie.douban.com'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = 'utf-8'
        return r.text

    def parse_html(self, _html):
        soup = BeautifulSoup(_html, 'lxml')
        urls = soup.find_all('div', class_='title')
        for url in urls:
            html = self.get_html_2(url.a['href'])
            self.get_content(html)

    def get_content(self, _html):
        soup = BeautifulSoup(_html, 'lxml')
        title = soup.find('span', property='v:itemreviewed')
        summary = soup.find('span', class_='all hidden')
        if summary is None:
            summary = soup.find('span', property='v:summary')
        if summary is None:
            return
        _summary = summary.text
        _summary = _summary.replace(' ', '')
        _summary = _summary.replace('　', '')
        _summary = _summary.replace('\n', '')
        directors = soup.find_all('a', rel='v:directedBy')
        actors = soup.find_all('a', rel='v:starring')
        kinds = soup.find_all('span', property='v:genre')
        lengths = soup.find_all('span', property='v:runtime')
        score = soup.find('strong', property='v:average')
        number = soup.find('span', property='v:votes')
        if not os.path.exists('m_movie'):
            os.mkdir('m_movie')
        self.index += 1
        with open('movie/' + str(self.index) + '.txt', 'w', encoding='utf-8') as file:
            file.write(title.text)
            file.write('\n')
            #file.write(_summary)
            #file.write('\n')
            for director in directors:
                file.write(director.text + ' ')
            #file.write('\n')
            for actor in actors:
                file.write(actor.text + ' ')
            #file.write('\n')
            for kind in kinds:
                file.write(kind.text + ' ')
            #file.write('\n')
            for length in lengths:
                file.write(length.text + ' ')
            #file.write('\n')
            file.write(score.text)
            #file.write('\n')
            file.write(number.text)

    def start(self):
        start = 0
        while start <= 1025:
            html = self.get_html_1('https://www.douban.com/doulist/13712700/?start=' + str(start) + '&sort=seq&sub_type=')
            self.parse_html(html)
            start += 25


spider = Spider()
spider.start()
