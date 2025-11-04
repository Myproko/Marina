import matplotlib.pyplot as plt
from collections import Counter


class Analysis:

    @staticmethod
    def views_by(json_entries: list, attr_value: str, filtering_attribute_name: str, views_by_attribute: str, draw_a_histogram = False, print_list = False):
        
        filtered_paragraphs = list(filter(lambda paragraph: paragraph.get(filtering_attribute_name) == attr_value, json_entries))
        
        attributes = []
        for filtered_paragraph in filtered_paragraphs:
            attributes.append(filtered_paragraph.get(views_by_attribute))
        
        count_attributes = Counter(attributes)
        
        if (draw_a_histogram):
            plt.style.use('ggplot')
            plt.title(f"Views by {views_by_attribute}")
            plt.bar(count_attributes.keys(), height = count_attributes.values(), align = 'center')
            plt.show()
        
        if(print_list):
            print(f"{views_by_attribute}s ", count_attributes.keys())
    