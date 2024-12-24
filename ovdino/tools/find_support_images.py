import json

def find_filename_by_support_ids(support_ids, annotations_file='support_annotations.json'):
    # Load the support_annotations.json file
    with open(annotations_file, 'r') as file:
        annotations_data = json.load(file)
    list_of_filenames = []
    # Iterate through the list of dictionaries
    for item in annotations_data:
        if item['category_id'] in support_ids:
            list_of_filenames.append("datasets/DatasetToBeEval/Support/images/"+item['filename'])
    return list_of_filenames
    
    # Return None if no match is found
    return None
def retrieve_annotation_by_filename(filename, annotations_file='support_annotations.json'):
    # Load the support_annotations.json file
    with open(annotations_file, 'r') as file:
        annotations_data = json.load(file)
    list_of_annotations = []
    # Iterate through the list of dictionaries
    for item in annotations_data:
        if item['filename'] == filename:
            list_of_annotations.append(item)
    return list_of_annotations
    
    # Return None if no match is found
    return None
if __name__ == "__main__":
    support_ids = [20, 21, 22, 23, 24]
    filename = find_filename_by_support_ids(support_ids)
    print(filename)
