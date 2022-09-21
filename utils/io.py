import os
import json
import yaml
import pickle


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = f.read()
        content = yaml.load(cfg, Loader=yaml.FullLoader)
    return content


def load_json(path: str):
    with open(path, 'r') as inf:
        return json.load(inf)


def load_jsons(root: dir):
    for path in os.listdir(root):
        if path.endswith('.json'):
            yield load_json(os.path.join(root, path))


def save_json(data: dict, path: str):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def save_dict(data: dict, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_dict(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def walker(root: dir):
    for base, _, files in os.walk(root):
        for f in files:
            yield os.path.join(base, f).replace('\\', '/').removeprefix(root)
