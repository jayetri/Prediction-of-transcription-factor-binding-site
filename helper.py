import numpy as np
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    SizeOFData = len(data)
    NOOfEpochsForEveryBatch = int((len(data)-1)/batch_size) + 1


    for e in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            INdShuff = np.random.permutation(np.arange(SizeOFData))
            shuffled_data = data[INdShuff]
        else:
            shuffled_data = data
        for batch_num in range(NOOfEpochsForEveryBatch):
            EndINd = min((batch_num + 1) * batch_size, SizeOFData)
            INdStart = batch_num * batch_size
            yield shuffled_data[INdStart:EndINd]