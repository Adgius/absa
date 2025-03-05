import xml.etree.ElementTree as ET
import polars as pl
import requests
import io

from tqdm import tqdm

def load_bytes(url):
    ds = requests.get(url)
    bytes_data = io.BytesIO(ds.content)
    return bytes_data

def get_data(path):
    tree = ET.parse(path)
    root = tree.getroot()
    id = []
    text = []
    target = []
    category = []
    polarity = []
    start = []
    end = []

    for sent in tqdm(root.findall('Review/sentences/sentence')):
        for asp in sent.findall('Opinions/Opinion'):
            id.append(sent.attrib['id'].split(':')[0])
            text.append(sent.find('text').text)
            target.append(asp.attrib['target'])
            category.append(asp.attrib['category'])
            polarity.append(asp.attrib['polarity'])
            start.append(asp.attrib['from'])
            end.append(asp.attrib['to'])
    df = {
            'id': id,
            'text': text,
            'target': target,
            'category': category,
            'polarity': polarity,
            'start': start,
            'end': end}
    return pl.DataFrame(df)