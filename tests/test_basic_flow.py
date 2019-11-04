from tempfile import NamedTemporaryFile

from hugin.io.loader import BinaryCategoricalConverter
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from hugin.infer.core import KerasPredictor

from hugin.infer.scene import RasterSceneTrainer

import os

def test_keras_train_complete_flow(generated_filesystem_loader):
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
        os.remove(named_temporary_file.name)
        keras_model = KerasPredictor(
            name='test_keras_trainer',
            model_path=named_temporary_file.name,
            model_builder="hugin.models.unet.unetv14:unet_v14",
             batch_size=50,
             epochs=1,
             metrics=[
                 "accuracy"
             ],
             loss="categorical_crossentropy",
             optimizer=Adam(),
             callbacks=[
                 EarlyStopping(monitor='loss')
             ]
        )
        trainer = RasterSceneTrainer(name="test_raster_trainer",
                                     stride_size=256,
                                     window_size=(256, 256),
                                     model=keras_model,
                                     mapping=mapping)

        dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        print("Training on %d datasets" % len(dataset_loader))
        print("Using %d datasets for validation" % len(validation_loader))

        trainer.train_scenes(dataset_loader, validation_scenes=validation_loader)
        trainer.save()

        assert os.path.exists(named_temporary_file.name)
        assert os.path.getsize(named_temporary_file.name) > 0