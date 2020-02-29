## this is an interface for all the dataset in this program

class Dataset:
    def get_batch_train(self, batch_size):
        raise NotImplementedError("the dataset is not implemented yet")

    def get_batch_validation(self):
        raise NotImplementedError("the dataset is not implemented yet")

    def get_batch_test(self):
        raise NotImplementedError("the dataset is not implemented yet")
