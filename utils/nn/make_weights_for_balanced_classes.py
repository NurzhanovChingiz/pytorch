import numpy as np
def make_weights_for_balanced_classes(classid):
    """
    From https://gist.github.com/ogvalt/59bda57ec014512e61fb2b16e4911f61
    """
    weight_per_class=np.bincount(classid)
    N=sum(weight_per_class)
    weight_per_class=N/weight_per_class.astype(float)
    weight_per_class[weight_per_class == np.inf] = 0                                 
    weight=weight_per_class[classid]                             
    return weight 
## Usage
# dataset_train = datasets.ImageFolder(traindir)                                                                         
                                                                                
# # For unbalanced dataset we create a weighted sampler                       
# weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
# weights = torch.DoubleTensor(weights)                                       
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,          

## Or
# train_paths, test_paths, train_labels, test_labels = train_test_split(img_paths_list, img_classes_list,stratify=img_classes_list, test_size=TEST_SIZE, random_state=BATCH_SIZE)
# train_dataset = ImageDataset(train_paths, train_labels, transform=transform_train)
# test_dataset = ImageDataset(test_paths, test_labels, transform=transform_test)
# weights = make_weights_for_balanced_classes(train_dataset, num_classes)                                                                
# weights = torch.DoubleTensor(weights)    
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                       