import requests
import os

def answer(name, answer):
    if not os.path.exists('./answers/'):
        os.makedirs('./answers/')
    with open('./answers/' + name, 'w+') as f:
        f.write(answer)

def load_file(name, url):
    try:
        with open(name) as f:
            data = ''.join(f.readlines())
    except FileNotFoundError:
        r = requests.get(url)
        data = r.text
        with open(name, 'w+') as f:
            f.write(data)
    return data