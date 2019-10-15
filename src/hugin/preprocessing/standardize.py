import pickle

class SkLearnStandardizer(object):
    def __init__(self, path):
        self.path = path
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)
    def __call__(self, band):
        return self.model.transform(band)
