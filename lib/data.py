import logging

import torch
from torchvision import datasets, transforms

from lib import sampler, utils


def _create_dataloader(dataset, is_train, batch_size, num_workers=0):
    """
    Create a dataloader for a given dataset.

    Args:
        dataset: torch.utils.data.Dataset used in the dataloader
        is_train: Bool indicating whether to generate a loader for training
        batch_size: Number of training samples per batch
        num_workers: Optional integer of how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0
    Returns:
        torch.utils.data.DataLoader
    """
    # For GPU acceleration store dataloader in pinned (page-locked) memory
    loader_kwargs = _get_loader_kwargs(num_workers)

    # Create dataloader objects
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train,
                                             drop_last=is_train, **loader_kwargs)

    return dataloader


def _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers):
    """
    Create multitask train- and testloaders.

    Args:
        train_datasets: list of torch.utils.data.Dataset for training
        test_datasets: list of torch.utils.data.Dataset for testing
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0
    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Combine all train datasets into a single big data set
    train_dataset_all = torch.utils.data.ConcatDataset(train_datasets)

    # Create a single big train loader containing all the tasks
    train_loader = _create_dataloader(train_dataset_all, True, batch_size, num_workers)

    # Create a list of data loaders for each test set
    test_loaders = [
        _create_dataloader(test, False, batch_size, num_workers)
        for test in test_datasets
    ]

    return train_loader, test_loaders


def _create_split_datasets(dataset, permutation):
    """
    Create a list of datasets split in pairs by target.

    Args:
        dataset: torch.utils.data.Dataset used in the dataloader
        permutation: Tensor of permuted indeces used to permute the images

    Returns:
        split_datasets_tasks: List of datasets grouped by label pairs
    """
    # Get the indices for samples from the different classes
    split_indices = [torch.nonzero(dataset.targets == label).squeeze() for label in range(10)]

    # Adapt the targets such that even task have a 0 label and odd tasks have 1 label
    dataset.targets = permutation[dataset.targets] % 2

    # Create Subsets of the whole dataset
    split_datasets = [torch.utils.data.Subset(dataset, indices) for indices in split_indices]

    # Shuffle tasks given specified permutation
    split_datasets = [split_datasets[i] for i in torch.argsort(permutation)]

    # Re-concatenate the datasets in pairs
    split_datasets_tasks = [
        torch.utils.data.ConcatDataset([even, odd])
        for even, odd in zip(split_datasets[:-1:2], split_datasets[1::2])
    ]

    return split_datasets_tasks


def _create_split_loader(dataset, is_train, batch_size, permutation, num_workers):
    """
    Create a dataloader for continual split task given a dataset.

    Args:
        dataset: torch.utils.data.Dataset to be split in pairs by target
        is_train: Bool indicating whether to generate a loader for training
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        List of torch.utils.data.DataLoader
    """
    # Re-concatenate the datasets in pairs
    split_datasets_tasks = _create_split_datasets(dataset, permutation)

    # Create the dataloader
    split_loaders = [
        _create_dataloader(dataset, is_train, batch_size, num_workers)
        for dataset in split_datasets_tasks
    ]

    return split_loaders


def _get_loader_kwargs(num_workers):
    """
    For GPU acceleration store dataloader in pinned (page-locked) memory

    Args:
        num_workers: Integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        Dictionary containing additional parameters for torch.utils.data.DataLoader
    """
    return {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {
        'num_workers': num_workers}


def _permute_tensor(input, permutation):
    """
    Permute elements of tensor given a matrix of permutation indices
    """
    # Cache the original dimensions
    dimensions = input.size()

    # Apply the permutation to the flattened tensor
    output_flat = torch.index_select(input.view(-1), 0, permutation)

    # Restore original dimensions
    output = output_flat.view(dimensions)
    #output = input.view(dimensions)

    return output


def create_blurry_perm_mnist_loader(num_tasks, epochs, batch_size, transition_portion, num_workers=0):
    """
    Create a single big dataloader for the blurry pMNIST-n task with multiple test loaders.

    Args:
        num_tasks: Number of permuted MNIST tasks to generate
        epochs: Number of epochs to train per task
        batch_size: Number of training samples per batch
        transition_portion: Float between 0 and 1 determining the portion
            of the task steps that transitions to the next task
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: Single big training torch.utils.data.DataLoader for all tasks and epochs
        test_loaders: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Generate pMNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_mnist_dataset(p, is_train=True) for p in permutations]
    test_datasets = [get_perm_mnist_dataset(p, is_train=False) for p in permutations]

    # Combine all train datasets into a single big data set
    train_dataset_all = torch.utils.data.ConcatDataset(train_datasets)

    # For GPU acceleration store dataloader in pinned (page-locked) memory
    loader_kwargs = _get_loader_kwargs(num_workers)

    # Create the sampling schedule in a blurry continual setting
    task_probs = sampler.create_task_probs(num_tasks, epochs * len(train_datasets[0]), transition_portion)
    train_sampler = sampler.ContinualMultinomialSampler(train_dataset_all, num_tasks, task_probs)

    # Create a single big train loader containing all the tasks
    train_loader = torch.utils.data.DataLoader(train_dataset_all, batch_size=batch_size,
                                               sampler=train_sampler, **loader_kwargs)

    # Create a list of data loaders for each test set
    test_loaders = [
        _create_dataloader(test, False, batch_size, num_workers)
        for test in test_datasets
    ]

    return train_loader, test_loaders


def create_fashion_mnist_loader(batch_size, num_workers=0):
    """
    Create train- and testloader for fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Get fashion MNIST
    train_dataset = get_fmnist_dataset(True)
    test_dataset  = get_fmnist_dataset(False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_mnist_loader(batch_size, num_workers=0):
    """
    Create a single train- and testloader for standard MNIST

    Args:
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Get permuted MNIST
    train_dataset = get_mnist_dataset(True)
    test_dataset  = get_mnist_dataset(False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_multitask_split_fmnist_loader(shuffle_tasks, batch_size, num_workers=0):
    """
    Create Multitask split fashion MNIST train- and testloader.

    Args:
        shuffle_tasks: Bool determining if task order should be shuffled
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split fashionMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    # Generate split fashionMNIST datsets
    train_datasets = _create_split_datasets(get_fmnist_dataset(True), permutation)
    test_datasets  = _create_split_datasets(get_fmnist_dataset(False), permutation)

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_split_mnist_loader(batch_size, shuffle_tasks, num_workers=0):
    """
    Create Multitask split MNIST train- and testloader.

    Args:
        shuffle_tasks: Bool determining if task order should be shuffled
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split MNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    # Generate split fashionMNIST datsets
    train_datasets = _create_split_datasets(get_mnist_dataset(True), permutation)
    test_datasets  = _create_split_datasets(get_mnist_dataset(False), permutation)

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_perm_fmnist_loader(num_tasks, batch_size, num_workers=0):
    """
    Create Multitask permuted fashion MNIST train- and testloader.

    Args:
        num_tasks: Number of permuted fashion MNIST tasks to generate
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Generate permuted fashion MNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_fmnist_dataset(p, is_train=True) for p in permutations]
    test_datasets  = [get_perm_fmnist_dataset(p, is_train=False) for p in permutations]

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_perm_mnist_loader(batch_size, num_tasks, num_workers=0):
    """
    Create Multitask permuted MNIST train- and testloader.

    Args:
        num_tasks: Number of permuted MNIST tasks to generate
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Generate pMNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_mnist_dataset(p, is_train=True) for p in permutations]
    test_datasets  = [get_perm_mnist_dataset(p, is_train=False) for p in permutations]

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_perm_fmnist_loader(batch_size, num_tasks, num_workers=0):
    """
    Create lists of train- and testloaders for permuted fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_tasks: Number of permuted fashion MNIST tasks to generate
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Create list of permutations
    permutations = [torch.randperm(28 * 28) for _ in range(num_tasks)]

    # Create a list of tuples of train, test loaders
    loaders = [
        create_perm_fmnist_loader_single(batch_size, p, num_workers)
        for p in permutations
    ]
    # Unpack the list of tuples into two separate lists
    train_loaders, test_loaders = map(list, zip(*loaders))

    return train_loaders, test_loaders


def create_perm_fmnist_loader_single(batch_size, permutation, num_workers=0):
    """
    Create a single train- and testloader for permuted fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: torch.utils.data.DataLoader of training data
        test_loader: torch.utils.data.DataLoader of test data
    """
    # Get permuted MNIST
    train_dataset = get_perm_fmnist_dataset(permutation, True)
    test_dataset  = get_perm_fmnist_dataset(permutation, False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_perm_mnist_loader(batch_size, num_tasks, num_workers=0):
    """
    Create lists of train- and testloaders for permuted MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_tasks: Number of permuted MNIST tasks to generate
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """

    # Create list of permutations
    permutations = [torch.randperm(28 * 28) for _ in range(num_tasks)] 

    # Create a list of tuples of train, test loaders
    loaders = [
        create_perm_mnist_loader_single(batch_size, p, num_workers)
        for p in permutations
    ]
    # Unpack the list of tuples into two separate lists
    train_loaders, test_loaders = map(list, zip(*loaders))
    

    return train_loaders, test_loaders, permutations


def create_perm_mnist_loader_single(batch_size, permutation, num_workers=0):
    """
    Create a single train- and testloader for permuted MNIST.

    Args:
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: torch.utils.data.DataLoader of training data
        test_loader: torch.utils.data.DataLoader of test data
    """
    # Get permuted MNIST
    train_dataset = get_perm_mnist_dataset(permutation, True)
    test_dataset  = get_perm_mnist_dataset(permutation, False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_split_mnist_loader(batch_size, shuffle_tasks, num_workers=0):
    """
    Create train and test_loaders for split MNIST.

    Args:
        batch_size: Number of training samples per batch
        shuffle_tasks: Bool determining if task order should be shuffled
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of torch.utils.data.DataLoader for all training tasks
        test_loader: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for splitMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    train_loader = _create_split_loader(get_mnist_dataset(True), True, batch_size, permutation, num_workers)
    test_loader  = _create_split_loader(get_mnist_dataset(False), False, batch_size, permutation, num_workers)

    return train_loader, test_loader


def create_split_fmnist_loader(batch_size, shuffle_tasks, num_workers=0):
    """
    Create train and test_loaders for split fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        shuffle_tasks: Bool determining if task order should be shuffled
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            Default: 0

    Returns:
        train_loader: List of torch.utils.data.DataLoader for all training tasks
        test_loader: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split fashionMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    train_loader = _create_split_loader(get_fmnist_dataset(True), True, batch_size, permutation, num_workers)
    test_loader  = _create_split_loader(get_fmnist_dataset(False), False, batch_size, permutation, num_workers)

    return train_loader, test_loader


def get_fmnist_dataset(is_train):
    """
    Create fashion MNIST data set using the letters split.

    """
    dataset = datasets.FashionMNIST('data/', train=is_train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.2861,), std=(0.3530,)),
                                    ]))

    return dataset


def get_mnist_dataset(is_train, random_subset = False):
    """
    Create MNIST data set without permutating the images.

    Args:
        is_train: Bool indicating whether to generate a loader for training

    Returns:
        dataset: MNIST torch.utils.data.Dataset
    """
    dataset = datasets.MNIST('data/', train=is_train, download=True, transform=transforms.Compose([
                             # Could insert transforms.Pad(2) here to obtain dimensions 32x32
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                             ]))
    dataset = datasets.MNIST('data/', train=is_train, download=True, transform=transforms.ToTensor())



    return dataset


def get_perm_mnist_dataset(permutation, is_train):
    """
    Create MNIST data set using the specified permutation on the images.
    Note: Validate using plt.imshow(mnist.data[0])

    Args:
        permutation: Tensor of permuted indeces used to permute the images
        is_train: Bool indicating whether to generate a loader for training

    Returns:
        dataset: permuted MNIST torch.utils.data.Dataset
    """
#     dkataset = datasets.MNIST('data/', train=is_train, download=True, transform=transforms.Compose([
# #                              # Could insert transforms.Pad(2) here to obtain dimensions 32x32
# #                              transforms.ToTensor(),
# #                              transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
# #                              lambda img: _permute_tensor(img, permutation),
# #                              ]))
    dataset = datasets.MNIST('data/', train=is_train, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda img: _permute_tensor(img, permutation),]))
    return dataset


def get_perm_fmnist_dataset(permutation, is_train):
    """
    Create MNIST data set using the specified permutation on the images.

    Args:
        permutation: Tensor of permuted indeces used to permute the images
        is_train: Bool indicating whether to generate a loader for training

    Returns:
        dataset: permuted fashion MNIST torch.utils.data.Dataset
    """
    dataset = datasets.FashionMNIST('data/', train=is_train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.2861,), std=(0.3530,)),
                                    lambda img: _permute_tensor(img, permutation),
                                    ]))
    return dataset
