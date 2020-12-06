

def load_data(path: str):
    with open(path, 'r') as f:
        result = f.readlines()
    return result
