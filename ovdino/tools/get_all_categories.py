import json

def get_categories_dicts(annotation_file):
  """
  Extracts categories from a COCO format annotation JSON file and creates 
  separate dictionaries for each category, mapping the category name to its ID.

  Args:
    annotation_file: Path to the annotation JSON file.

  Returns:
    A list of dictionaries, where each dictionary represents a category 
    and contains a single key-value pair: the category name and its ID.
  """

  with open(annotation_file, 'r') as f:
    data = json.load(f)

  categories = data['categories']
  category_dicts = [{"name":category['name'],"id":category['id']} for category in categories]
  return category_dicts

# Example usage
annotation_file = '/root/workspace/ZladWu/OV-DINO/ovdino/datasets/OpenImages1500K/labels.json'  # Replace with your file path
category_dicts = get_categories_dicts(annotation_file)
#save as json
with open('category_dicts.json', 'w') as f:
  json.dump(category_dicts, f, indent=1)
print(len(category_dicts))