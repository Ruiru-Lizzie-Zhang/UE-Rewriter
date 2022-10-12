def read_json(directory):
    
    import json
    # read data as list of list, each smaller list is a dialog composing of X sentences (8<=X<=23 for Wizard of Wikipedia)
    
    docs = []
    f = open(directory)
    data = json.load(f)
    for i in data:
        docs.append(i)
    f.close()

    all_data = []
    for doc in docs:
        dialog = [i['text'].lower() for i in doc['dialog']]
        all_data.append(dialog)
    return all_data


def read_txt(directory):
    
    # read data as list of sentences, separated by ##
    
    file = open(directory, 'r', encoding='utf-8')
    file_data = file.read() 
    all_data = file_data.split('\n')

    while "" in all_data:
        all_data.remove("")
    return all_data