import json
import requests
import matplotlib.pyplot as plt
from collections import Counter

class Analysis:
    @staticmethod
    def views_by(json_entries: list, attr_value: str, filtering_attribute_name: str, views_by_attribute: str,
                 draw_a_histogram=False, print_list=False):
        
        filtered_paragraphs = [
            p for p in json_entries
            if str(p.get(filtering_attribute_name, "")).strip() == attr_value.strip()
        ]
        
        print(f"Отфильтровано записей: {len(filtered_paragraphs)}")

        attributes = [fp.get(views_by_attribute) for fp in filtered_paragraphs]
        count_attributes = Counter(attributes)

        if draw_a_histogram:
            if count_attributes:
                plt.style.use('ggplot')
                plt.title(f"Views by {views_by_attribute}")
                plt.bar(count_attributes.keys(), count_attributes.values(), align='center')
                plt.xticks(rotation=45)
                plt.show()
            else:
                print("⚠️ Нет данных для построения графика.")
        
        if print_list:
            print(f"{views_by_attribute}s:", list(count_attributes.keys()))

# --- Загрузка JSON ---
url = "https://www.macs.hw.ac.uk/~hwloidl/Courses/F21SC/issuu_sample.json"
response = requests.get(url, verify=False)

paragraphs = []
for line in response.text.strip().splitlines():
    if line.strip():
        try:
            paragraphs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print("Всего записей:", len(paragraphs))
print("Example first paragraph:", paragraphs[0])

# --- Автоматический поиск всех env_doc_id и visitor_uuid ---
all_env_doc_ids = sorted({p.get("env_doc_id") for p in paragraphs if p.get("env_doc_id")})
all_visitor_uuids = sorted({p.get("visitor_uuid") for p in paragraphs if p.get("visitor_uuid")})

print("\nПримеры существующих env_doc_id:", all_env_doc_ids[:5])
print("Примеры существующих visitor_uuid:", all_visitor_uuids[:5])

# --- Примеры использования ---
# 1️⃣ Views by country для первого существующего env_doc_id
plt.style.use('default')
Analysis.views_by(
    json_entries=paragraphs,
    attr_value=all_env_doc_ids[0],
    filtering_attribute_name='env_doc_id',
    views_by_attribute='visitor_country',
    draw_a_histogram=True
)

# 2️⃣ Documents viewed by первого существующего visitor_uuid
Analysis.views_by(
    json_entries=paragraphs,
    attr_value=all_visitor_uuids[0],
    filtering_attribute_name='visitor_uuid',
    views_by_attribute='env_doc_id',
    print_list=True
)

# 3️⃣ Users who viewed первый существующий env_doc_id
Analysis.views_by(
    json_entries=paragraphs,
    attr_value=all_env_doc_ids[0],
    filtering_attribute_name='env_doc_id',
    views_by_attribute='visitor_uuid',
    print_list=True
)
