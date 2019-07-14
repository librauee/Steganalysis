# -*- coding: utf-8 -*-


"""
阿里文学： 章节url里的sign参数是服务器返回的固定参数, 不是本地加密参数, 在小说内容页源码里
"""


import re
import json
from bs4 import BeautifulSoup
from requests import Session
import execjs

class book_downloader:

    def __init__(self):
        self.bookId = input('请输入小说Id>> \n')
        self.session = Session()
        self.session.headers_ = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36'
        }
        self.book_url = 'https://www.aliwx.com.cn/reader?bid={}'.format(self.bookId)
        self.chapter_url = 'https://c13.shuqireader.com/pcapi/chapter/contentfree/'
        with open('test.txt','rb') as f:
            js = f.read().decode()
        self.ctx = execjs.compile(js)

    def _get_book_info(self):
        """
        小说的所有信息, 包括所有章节
        :return:
        """
        res = self.session.get(self.book_url)
        bsobj = BeautifulSoup(res.text, 'lxml')
        book_info = bsobj.find('i', {'class': 'page-data js-dataChapters'}).get_text()
        return json.loads(book_info)

    def _parse_book(self, book_info):
        """
        解析获取小说基本信息和章节列表
        :param book_info:
        :return:
        """
        bookName = book_info.get('bookName', '')
        authorName = book_info.get('authorName', '')
        chapterNum = book_info.get('chapterNum', 0)
        bookCover = book_info.get('imgUrl', '')
        wordCount = book_info.get('wordCount', 0)
        contentUrl = book_info.get('contentUrl', '')
        book_data = {
            'bookName': bookName,
            'authorName': authorName,
            'chapterNum': chapterNum,
            'bookCover': bookCover,
            'wordCount': wordCount,
            'contentUrl': contentUrl,
        }
        print(book_data)
        chapterList = book_info['chapterList'][0]['volumeList']
        chapter_list = []
        for chapter in chapterList:
            chapterName = chapter.get('chapterName', '')
            chapterPrice = chapter.get('chapterPrice', 0)
            chapterWordCount = chapter.get('wordCount', 0)
            chapterUrlSuffix = chapter.get('contUrlSuffix', '')
            chapter_data = {
                'chapterName': chapterName,
                'chapterPrice': chapterPrice,
                'chapterWordCount': chapterWordCount,
                'chapterUrlSuffix': chapterUrlSuffix
            }
            chapter_list.append(chapter_data)
        return chapter_list,bookName

    def _get_encrypt_content(self, chapterUrlSuffix):
        """
        请求章节接口获取加密内容
        :param chapterUrlSuffix:
        :return:
        """
        chapter_id = re.search('chapterId=(\d+)', chapterUrlSuffix).group(1)
        self.session.headers.update({
            'Referer': f'https://www.aliwx.com.cn/reader?bid={self.bookId}&cid={chapter_id}'
        })
        url = self.chapter_url + chapterUrlSuffix
        res = self.session.get(url).json()
        if res['message'] == "success":
            return res['ChapterContent']
        else:
            print((res['message']))
            return False

    def _decrypt(self, content):
        """
        解密
        :param content:
        :return:
        """
        return self.ctx.call('_decodeCont', content)

    def run(self):
        book_info = self._get_book_info()
        chapter_list ,book_Name= self._parse_book(book_info)
        with open('{}.txt'.format(book_Name),'w') as f:
            for chapter in chapter_list:
                if chapter['chapterPrice'] == 0:
                    chapter_name = chapter['chapterName']
                    encrypt_content = self._get_encrypt_content(chapter['chapterUrlSuffix'])
                    if encrypt_content:
                        chapter_content = self._decrypt(encrypt_content).replace('<br/>', '\n')
                        f.write('\n---------------章节分界线--------------\n'+chapter_name)
                        f.write(chapter_content)
                    
                else:
                    print('你该充钱啦! 不充钱怎么变强！！！')
                    break

if __name__ == '__main__':
    spider = book_downloader()
    spider.run()
