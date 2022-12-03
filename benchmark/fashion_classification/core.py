from torchvision import datasets, transforms
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import XYTaskPipe as TaskPipe
class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, selected_labels = [0,2,6], seed=0):
        super(TaskGen, self).__init__(benchmark='fashion_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/FASHION',
                                      seed=seed
                                      )
        self.num_classes = len(selected_labels)
        self.selected_labels = selected_labels
        self.label_dict = {0: 'T-shirt', 1: 'Trouser', 2: 'pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Abkle boot'}
        self.save_task = TaskPipe.save_task

    def load_data(self):
        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i
        self.train_data = datasets.FashionMNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.FashionMNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        self.train_data = TaskPipe.TaskDataset(train_data_x, train_data_y)
        test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        self.test_data = {'x':test_data_x, 'y':test_data_y}

    def convert_data_for_saving(self):
        train_x, train_y = self.train_data.tolist()
        self.train_data = {'x':train_x, 'y':train_y}
        return

    def save_task(self, generator):
        self.convert_data_for_saving()
        XYTaskPipe.save_task(self)
