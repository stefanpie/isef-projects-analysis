from itertools import groupby
import bs4
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
import json
from pprint import pprint as pp
import re
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns


with open("./data.json", "r", encoding="utf8") as f:
    input_file_data = f.read()

input_json = json.loads(input_file_data)
input_json.pop('context', None)
input_json.pop('length', None)
input_json.pop('selector', None)
input_json.pop('ajax', None)


all_abstracts_df = pd.DataFrame.from_dict(input_json, orient="index")
all_abstracts_df.columns = ["year", "name", "project_title",
                            "category", "fair_country", "fair_state", "fair_province", "awards"]

all_abstracts_df['year'] = all_abstracts_df['year'].astype(int)
all_abstracts_df['name'] = all_abstracts_df['name'].astype("string")
all_abstracts_df['project_title'] = all_abstracts_df['project_title'].astype("string")
all_abstracts_df['category'] = all_abstracts_df['category'].astype("string")
all_abstracts_df['fair_country'] = all_abstracts_df['fair_country'].astype("string")
all_abstracts_df['fair_state'] = all_abstracts_df['fair_state'].astype("string")
all_abstracts_df['fair_province'] = all_abstracts_df['fair_province'].astype("string")
all_abstracts_df['awards'] = all_abstracts_df['awards'].astype("string")


pattern_title = re.compile(r"<a.*>(.*)</a>")
pattern_link = re.compile(r"<a href=\"(.*)\">")
base_url = "https://abstracts.societyforscience.org"


def process_title_abstract_link(title_link):
    title_link = title_link.replace("\n", "")

    title = re.match(pattern_title, title_link).group(1)

    link = re.match(pattern_link, title_link).group(1)
    link = base_url + link
    link_path, link_params = link.split("?")
    link_project_id_param = link_params.split(";")[-1]
    link_working = link_path+"?"+link_project_id_param

    # r = requests.get(link_working)
    # abtract_page = r.text
    # abtract_page_soup = BeautifulSoup(abtract_page, 'html.parser')
    # abstract = abtract_page_soup.select("body > div > div.container > div > div > p:nth-child(6)")[0].get_text()
    # abstract = abstract.replace("Abstract:", "")
    # abstract = " ".join(abstract.split())

    # print(title)
    # print(abstract)
    # print()
    return title, link_working


all_abstracts_df['project_title'], all_abstracts_df['project_abstract_link'] = zip(
    *all_abstracts_df['project_title'].apply(process_title_abstract_link))

all_abstracts_df['project_title'] = all_abstracts_df['project_title'].astype("string")
all_abstracts_df['project_abstract_link'] = all_abstracts_df['project_abstract_link'].astype("string")


def process_awards(award):
    if award == "":
        return []
    else:
        award_list = award.split("<br>")
        award_list = [a.strip() for a in award_list]
        return award_list


all_abstracts_df["awards"] = all_abstracts_df["awards"].apply(process_awards)

# pd.set_option('display.max_rows', None)


all_abstracts_df["won_awards"] = all_abstracts_df["awards"].apply(lambda al: len(al) > 0)
all_abstracts_df["num_awards"] = all_abstracts_df["awards"].apply(lambda al: len(al))

x = all_abstracts_df["project_title"].to_numpy()
y = all_abstracts_df["won_awards"].to_numpy().astype(int)

all_abstracts_df.to_csv("processed_data.csv", index=True)