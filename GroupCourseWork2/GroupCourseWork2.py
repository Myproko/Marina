import json
import requests
import pandas as pd
import matplotlib as mp
import tkinter as tk
from entry import Entry
url = "https://www.macs.hw.ac.uk/~hwloidl/Courses/F21SC/issuu_sample.json"
response = requests.get(url,verify = False)
data = response.content.decode("utf-8")
data_list=data.split("}")

paragraphs = response.content.decode("utf-8")
paragraphs = paragraphs.strip().split("\n")
paragraphs = [json.loads(paragraph) for paragraph in paragraphs]

#testing our views_by. It takes
#1. an attribute value
#2. a filtering attribute name
#3. the attribute that it needs to search for

#views by country/continent
Analysis.views_by(
    json_entries = paragraphs, 
    attr_value = '131224090853-45a33eba6ddf71f348aef7557a86ca5f', 
    filtering_attribute_name = 'env_doc_id', 
    views_by_attribute = 'visitor_country',
    draw_a_histogram = True
)
#all documents viewed by user
Analysis.views_by(
    json_entries = paragraphs, 
    attr_value = 'f2e00a44114b4b0d', 
    filtering_attribute_name = 'visitor_uuid', 
    views_by_attribute = 'env_doc_id',
    print_list = True
)

#all users that viewed a document
Analysis.views_by(
    json_entries = paragraphs, 
    attr_value = '140205141802-000000007cf86ca1250f9fb5fbfa8102', 
    filtering_attribute_name = 'env_doc_id', 
    views_by_attribute = 'visitor_uuid',
    print_list = True
)
