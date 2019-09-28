import logging

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

log = logging.getLogger(__name__)

def train_handler(config, args):
    input_dir = args.input_dir
    data_source = config["data_source"]
    predictor = config["trainer"]
    if input_dir is not None:
        data_source.input_source = input_dir

    log.info("Using datasource: %s", data_source)
    log.info("Atempting to traing with data in %s", data_source.input_source)

    dataset_loader, validation_loader = data_source.get_dataset_loaders()
    log.info("Training on %d datasets", len(dataset_loader))
    log.info("Using %d datasets for validation", len(validation_loader))

    dataset_loader.loop = True
    validation_loader.loop = True
    predictor.train_scenes(dataset_loader, validation_scenes=validation_loader)