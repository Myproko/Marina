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
for paragraph in data_list:
    paragraph = paragraph + "}"
    paragraph = json.dumps(paragraph, indent = 4)
    paragraph = json.loads(paragraph)
    current_entry = Entry(ts = "test", visitor_uuid = "1234", visitor_source = "russia")


