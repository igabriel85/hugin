import os
from tempfile import NamedTemporaryFile

import pytest

from hugin.infer.core import IdentityModel
from hugin.infer.scene import RasterSceneTrainer
from hugin.io.loader import BinaryCategoricalConverter
from tests.conftest import generate_filesystem_loader


@pytest.fixture
def small_generated_filesystem_loader():
    return generate_filesystem_loader(num_images=4, width=500, height=510)

#@pytest.mark.skipif(not runningInCI(), reason="Skipping running locally as it might be too slow")
def test_keras_train_complete_flow(generated_filesystem_loader, small_generated_filesystem_loader):
    mapping = {
        'inputs': {
            'input_1': {
                'primary': True,
                'channels': [
                    ["RGB", 1]
                ]
            }
        },
        'target': {
            'output_1': {
                'channels': [
                    ["GTI", 1]
                ],
                'window_shape': (256, 256),
                'stride': 256,
                'preprocessing': [
                    BinaryCategoricalConverter(do_categorical=False)
                ]
            }
        }
    }

    with NamedTemporaryFile(delete=False) as named_temporary_file:
        named_tmp = named_temporary_file.name
        os.remove(named_temporary_file.name)
        identity_model = IdentityModel(name="dummy_identity_model", num_loops=3)
        trainer = RasterSceneTrainer(name="test_raster_trainer",
                                     stride_size=256,
                                     window_size=(256, 256),
                                     model=identity_model,
                                     mapping=mapping,
                                     destination=named_tmp)

        dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        loop_dataset_loader_old = dataset_loader.loop
        loop_validation_loader_old = validation_loader.loop

        try:
            dataset_loader.loop = True
            validation_loader.loop = True
            print("Training on %d datasets" % len(dataset_loader))
            print("Using %d datasets for validation" % len(validation_loader))

            trainer.train_scenes(dataset_loader, validation_scenes=validation_loader)
            trainer.save()

            assert os.path.exists(named_tmp)
            assert os.path.getsize(named_tmp) > 0
        finally:
            dataset_loader.loop = loop_dataset_loader_old
            validation_loader.loop = loop_validation_loader_old
