
import numpy as np
import torch
from dataset.mydataset import MyTrainDataset,MyTestDataset

def get_dataset(args, transform_train1, transform_train2,transform_test,k):
    # prepare datasets
    xtrain = np.load(f"{args.dataset}/x_train_fold{k}.npy",allow_pickle=True)
    print(xtrain.shape)
    ytrain =torch.Tensor(np.load(f"{args.dataset}/y_train_fold{k}.npy",allow_pickle=True)).view(-1)
    xtest = np.load(f"{args.dataset}/x_test_fold{k}.npy",allow_pickle=True)
    ytest =torch.Tensor(np.load(f"{args.dataset}/y_test_fold{k}.npy",allow_pickle=True)).view(-1)
    # normalize
    mean_per_channel = np.mean(xtrain, axis=(0, 2), keepdims=True)  
    std_per_channel = np.std(xtrain, axis=(0, 2), keepdims=True)
    data_normalized = (xtrain - mean_per_channel) / std_per_channel
    xtrain=torch.Tensor(data_normalized).unsqueeze(1).float()

    mean_per_channel = np.mean(xtest, axis=(0, 2), keepdims=True) 
    std_per_channel = np.std(xtest, axis=(0, 2), keepdims=True)
    data_normalized = (xtest - mean_per_channel) / std_per_channel
    xtest=torch.Tensor(data_normalized).unsqueeze(1).float()
    
    #################################### Unreliable Annotation Train set #############################################
    trainset = UATrain(args, xtrain,ytrain, transform_train1, transform_train2,transform_test)

    trainset.random_unreliable_annotation()

    trainset.labelsNoisyOriginal = trainset.targets.copy()

    #################################### Test set #############################################
    testset = MyTestDataset(xtest,ytest,transform_test)

    return trainset, testset, trainset.clean_labels, trainset.noisy_labels, trainset.noisy_indexes,  trainset.labelsNoisyOriginal

class UATrain(MyTrainDataset):
    def __init__(self, args, data,labels,transform1=None, transform2=None,target_transform=None):
        super(UATrain, self).__init__(data,labels)
        self.transform1 = transform1
        self.transform2 = transform2

        self.target_transform = target_transform

        self.args = args
       

        self.num_classes = self.args.num_classes
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []

        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.ua_ratio)


   
    def random_unreliable_annotation(self):

        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.int64)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
      

    def __getitem__(self, index):
      
        eeg, labels, soft_labels, noisy_labels, clean_labels = self.data[index], self.targets[index], self.soft_labels[
            index], self.labelsNoisyOriginal[index], self.clean_labels[index]
        eeg_noDA = eeg

        eeg1 = self.transform1(eeg)
        eeg2 = self.transform1(eeg)
        
        return eeg1, eeg2, eeg_noDA, labels, soft_labels, index, noisy_labels, clean_labels

       