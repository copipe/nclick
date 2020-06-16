import dill


class IO:
    @classmethod
    def dump_pickle(cls, obj, path):
        with open(path, 'wb') as f:
            dill.dump(obj, f)

    @classmethod
    def load_pickle(cls, obj, path):
        with open(path, 'rb') as f:
            return dill.load(f)