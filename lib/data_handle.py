

en_path = '../data/korean-english-park.train/korean-english-park.train.en'
ko_path = '../data/korean-english-park.train/korean-english-park.train.ko'


def load_data(path: str):
    with open(path, 'r') as f:
        result = f.readlines()
    return result


if __name__ == '__main__':
    load_data(en_path)