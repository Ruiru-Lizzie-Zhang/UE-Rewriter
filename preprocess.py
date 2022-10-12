def read_data(directory):
    
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
        all_data.append(dialog_lower)
    return all_data