import random

import torch


class ContinualMultinomialSampler(torch.utils.data.Sampler):
    """
    Samples elements in a continual setting given the multinomial
    probability distribution specified in task_probs.
    """

    def __init__(self, data_source, num_tasks, task_probs):
        """
        Args:
            data_source: torch.utils.data.Dataset to sample from
            num_tasks: Number of tasks contained in the dataset (equal size is assumed)
            task_probs: Multinomial probability distribution specifying the probability of a sample for each iteration in a new row
        """
        self.data_source = data_source
        self.num_tasks = num_tasks
        self.task_probs = task_probs

    def __iter__(self):
        # Determine the number of samples per task (assuming equal taks sizes)
        task_samples = len(self.data_source) / self.num_tasks

        # Determine the range of indeces per task and store in tuples
        task_ranges = [
            (t * task_samples, (t + 1) * task_samples - 1) for t in range(self.num_tasks)
        ]

        # Sample the task id per iteration
        sampled_tasks = torch.multinomial(self.task_probs, 1).squeeze()

        # Sample the sample indeces depending on the sampled tasks
        sample_indeces = [random.randint(*task_ranges[task_id]) for task_id in sampled_tasks]

        return iter(sample_indeces)

    def __len__(self):
        # The number of samples is implictly defined by the dimension of task_probs
        return self.task_probs.size(0)


def create_task_probs(num_tasks, task_steps, transition_portion):
    """
    Create the task probabilities over the course of the whole continual task.
    NOTE: After the last task we transition to the first task again to balance
    the number of samples and transitions seen by every task

    Args:
        num_tasks: Number of tasks
        task_steps: Steps to be taken per task
        transition_portion: Portion of the task steps that transitions to the next task in [0,1]
    """

    # Compute the number of transition steps
    transition_steps = int(task_steps * transition_portion)

    # Initialize the whole tensor of evolving task probabilities with zeros
    probs_task = torch.zeros((task_steps * num_tasks, num_tasks))

    for task_id in range(num_tasks):
        # Create the deterministic section where only a single task is presented as a one-hot encoding
        probs_single = torch.zeros((task_steps - transition_steps, num_tasks))
        probs_single[:, task_id] = 1.0

        # Create the transition probabilities with a linear transition between two classes
        probs_transition = torch.zeros((transition_steps, num_tasks))
        probs_transition[:, task_id] = 1 - torch.linspace(0, 1, steps=transition_steps)
        probs_transition[:, (task_id + 1) % num_tasks] = torch.linspace(0, 1, steps=transition_steps)

        # Concatenate the two phases of probabilities to a single tensor
        probs_task[task_id * task_steps:(task_id + 1) * task_steps, :] = torch.cat((probs_single, probs_transition))

    # To make task comparable to Zeno 2019 remove last transition and add samples for first and last task
    probs_task[-transition_steps:, :] = torch.zeros((transition_steps, num_tasks))
    probs_task[-transition_steps:, num_tasks - 1] = 1.0
    probs_task = torch.cat((torch.zeros((transition_steps, num_tasks)), probs_task))
    probs_task[:task_steps - transition_steps, 0] = 1.0

    return probs_task
