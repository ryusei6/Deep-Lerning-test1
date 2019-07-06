import os
import json
import urllib
import argparse
import requests
from bs4 import BeautifulSoup

class Google(object):
    def __init__(self):
        self.url = "https://www.google.co.jp/search"
        self.session = requests.session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) \
                    Gecko/20100101 Firefox/10.0"
            }
        )

    def search(self, keyword, max):
        print('searching \"{}\"...'.format(keyword))
        query_generator = self._query_generator(keyword)
        return self._image_search(query_generator, max)

    def _query_generator(self, keyword):
        page = 0
        while page<1:
            params = urllib.parse.urlencode(
                {"q": keyword, "tbm": "isch", "ijn": str(page)}
            )
            yield self.url + "?" + params
            page += 1

    def _image_search(self, query_generator, max):
        results = []
        total = 0
        while True:
            # get url
            html = self.session.get(next(query_generator)).text
            soup = BeautifulSoup(html, "lxml")
            elements = soup.select(".rg_meta.notranslate")
            jsons = [json.loads(e.get_text()) for e in elements]
            image_url_list = [js["ou"] for js in jsons]

            # add search results
            if not len(image_url_list):
                print("-> No more images")
                break
            elif len(image_url_list) > max - total:
                results += image_url_list[: max - total]
                break
            else:
                results += image_url_list
                total += len(image_url_list)

        print("-> Found", str(len(results)), "images")
        return results


def input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", help="target name", type=str, required=True)
    parser.add_argument("-n", "--number", help="number of images", type=int, required=True)
    parser.add_argument("-d", "--directory", help="download location", type=str, default="./imgs")
    args = parser.parse_args()

    imgs_dir = args.directory
    target_name = args.target
    number = args.number

    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(os.path.join(imgs_dir, target_name), exist_ok=True)

    return {'imgs_dir':imgs_dir, 'target_name':target_name, 'number':number}

def download(results,imgs_dir,target_name):
    download_errors = []
    for i, url in enumerate(results):
        print("-> Downloading image", str(i + 1).zfill(4), end=" ")
        try:
            urllib.request.urlretrieve(
                url,
                os.path.join(imgs_dir, target_name, str(i + 1).zfill(4) + ".jpg"),
            )
            print("successful")
        except BaseException:
            print("failed")
            download_errors.append(i + 1)
            continue

    print("-" * 50)
    print("Complete downloaded")
    print("├─ Successful downloaded", len(results) - len(download_errors), "images")
    print("└─ Failed to download", len(download_errors), "images", *download_errors)


def main():
    input_data = input()
    google = Google()
    results = google.search(input_data['target_name'], input_data['number'])
    download(results,input_data['imgs_dir'],input_data['target_name'])

if __name__ == '__main__':
    main()
