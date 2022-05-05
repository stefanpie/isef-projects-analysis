from itertools import groupby
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
from pprint import pprint as pp
from bs4 import BeautifulSoup
import requests
from joblib import delayed, Parallel

df = pd.read_csv("./processed_data.csv", index_col=0)

idx = df.index.tolist()
links = df["project_abstract_link"].tolist()


def get_single_abstract(link):
    r = requests.get(link)
    abtract_page = r.text
    abtract_page_soup = BeautifulSoup(abtract_page, 'html.parser')
    abstract = abtract_page_soup.select("body > div > div.container > div > div > p:nth-child(6)")[0].get_text()
    abstract = abstract.replace("Abstract:", "")
    abstract = " ".join(abstract.split())


abstracts = Parallel(n_jobs=-1, verbose=11)(delayed(get_single_abstract)(l) for l in links)
