import logging
import os
import time
from importlib import reload

import torch
from torch.utils.tensorboard import SummaryWriter

# Global variables
log_dir = None
writer = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logging(name, dir=""):
    """
    Setup the logging device to log into a uniquely created directory
    """
    # Setup global logging directory
    global log_dir
    log_dir = os.path.join("log", dir, name)

    # Create the logging folder if it does not exist already
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Need to reload logging as otherwise the logger might be captured by another library
    reload(logging)

    # Setup global logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-5.5s %(asctime)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, name + "_event.log")),
            logging.StreamHandler()
        ])

    # Setup tensorboard writer device
    global writer
    writer = SummaryWriter(os.path.join(log_dir, name + "_tensorboard"))
