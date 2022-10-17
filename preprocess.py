def read_json(directory, clean=True):
    
    import json
    # read data as list of list, each smaller list is a dialog composing of X sentences (8<=X<=23 for Wizard of Wikipedia)
    
    f = open(directory)
    docs = list(json.load(f))
    f.close()

    all_data = []
    if clean:
        import re
        for doc in docs:
            dialog = [re.sub(r"[,.;@#?!&$/]+\ *", " ", i['text']).lower() for i in doc['dialog']]
            all_data.append(dialog)
    else:
        for doc in docs:
            dialog = [i['text'].lower() for i in doc['dialog']]
            all_data.append(dialog)        
    return all_data


def read_txt(directory):
    
    # read data as list of CLEANED LOWER-CASE sentences, separated by ##
    
    file = open(directory, 'r', encoding='utf-8')
    file_data = file.read() 
    all_data = file_data.split('\n')

    while "" in all_data:
        all_data.remove("")
    return all_data
