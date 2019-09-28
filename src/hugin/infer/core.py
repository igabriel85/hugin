from logging import getLogger

import numpy as np
import os

from hugin.tools.utils import import_model_builder

log = getLogger(__name__)

def metric_processor(func):
    def __metric_handler(self, scene, *args, **kwargs):
        metrics = kwargs.pop('_metrics', {})
        original_result = func(self, scene, *args, _metrics=metrics, **kwargs)
        scene_id, scene_data = scene
        my_metrics = {}
        if self.name not in metrics:
            metrics[self.name] = my_metrics
        else:
            my_metrics = metrics[self.name]
        if self.metrics:
            if self.gti_component in scene_data:
                gti = scene_data[self.gti_component]
                for metric_name, metric_calculator in self.metrics.items():
                    my_metrics[metric_name] = metric_calculator(original_result, gti.read())
            else:
                log.warning("Missing GTI data for GTI component: %s", self.gti_component)
        return (original_result, metrics)

    return __metric_handler
def postprocessor(func):
    def __postprocessor_handler(self, *args, **kwargs):
        kwargs.pop("_metrics", None)
        result = func(self, *args, **kwargs)
        postprocessors = getattr(self, 'post_processors')
        if not postprocessors:
            log.debug("No post-processors found for: %s", self)

        if callable(postprocessors):
            return postprocessors(result)

        try:
            iter(postprocessors)
            isiter = True
        except TypeError:
            isiter = False

        if isiter:
            for processor in postprocessors:
                if callable(processor):
                    log.debug("Running processor %s", processor)
                    result = processor(result)
                else:
                    log.warning("Non-Callable processor %s", processor)
            return result
        else:
            log.debug("No valid post-processors found for: %s", self)
            return result
    return __postprocessor_handler

def identity_processor(arg):
    return arg

class CategoricalConverter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, probability_array):
        prediction = np.argmax(probability_array, -1).astype(np.uint8)
        return prediction.reshape(prediction.shape + (1,))

class RasterModel(object):
    def __init__(self,
                 name=None,
                 batch_size=1,
                 swap_axes=True,
                 input_shapes=None,
                 output_shapes=None,
                 #input_shape=None,
                 #output_shape=None
                 ):
        """Base model object handling prediction

        :param name: Name of the model (optional)
        :param batch_size: Batch size used by `predict_scene_proba` when sending data `predict`. Default: 1
        :param swap_axes: Swap input data axes
        :param input_shape: Window size to be used by `predict_scene_proba`
        """
        instance_path = ".".join([self.__module__, self.__class__.__name__])
        self.name = "%s[%s]" % (instance_path, name if name is not None else self.__hash__())

        self.batch_size = batch_size
        self.swap_axes = swap_axes
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        #self.input_shape = input_shape
        #self.output_shape = output_shape

    def predict(self, batch):
        """Runs the predictor on the input tile batch

        :param batch: The input batch
        :return: returns a prediction according to model configuration
        """
        raise NotImplementedError()

    def save_model(self, destination=None):
        raise NotImplementedError()


class PredictionMerger(object):
    def __init__(self, height, width, depth, dtype):
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype

    def update(self, xstart, xend, ystart, yend, prediction):
        raise NotImplementedError()

    def get_prediction(self):
        raise NotImplementedError()


class NullMerger(PredictionMerger):
    def __init__(self, *args, **kwargs):
        super(NullMerger, self).__init__(*args, **kwargs)


class AverageMerger(PredictionMerger):
    def __init__(self, *args, **kwargs):
        super(AverageMerger, self).__init__(*args, **kwargs)
        self.tile_array = np.zeros((self.width, self.height, self.depth), dtype=self.dtype)
        self.tile_freq_array = np.zeros((self.width, self.height), dtype=np.uint8)

    def update(self, xstart, xend, ystart, yend, prediction):
        self.tile_array[xstart: xend, ystart: yend] += prediction
        self.tile_freq_array[xstart: xend, ystart: yend] += 1

    def get_prediction(self):
        tile_array = self.tile_array.copy()
        channels = tile_array.shape[-1]
        unique, counts = np.unique(self.tile_freq_array, return_counts=True)
        print(dict(zip(unique, counts)))
        for i in range(0, channels):
            tile_array[:, :, i] = tile_array[:, :, i] / self.tile_freq_array
        return tile_array


class KerasPredictor(RasterModel):
    def __init__(self,
                 model_path,
                 model_builder,
                 *args,
                 optimizer=None,
                 loss=None,
                 model_builder_options={},
                 epochs=1,
                 verbose=1,
                 callbacks=None,
                 validation_freq=1,
                 class_weight=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 shuffle=True,
                 initial_epoch=0,
                 steps_per_epoch=None,
                 validation_steps_per_epoch=None,
                 load_only_weights=False,
                 input_shape=None,
                 custom_objects={},
                 **kwargs):
        RasterModel.__init__(self, *args, **kwargs)
        self.custom_objects = custom_objects
        self.model_path = model_path
        self.load_only_weights = load_only_weights
        self.model_builder_options = model_builder_options
        if 'input_shapes' not in self.model_builder_options:
            self.model_builder_options.update(input_shapes = self.input_shapes)
        if 'output_shapes' not in self.model_builder_options:
            self.model_builder_options.update(output_shapes = self.output_shapes)
        self.model_builder = model_builder
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = []
        self.validation_freq = validation_freq
        self.class_weight = class_weight
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.shuffle = shuffle
        self.initial_epoch = initial_epoch
        self.optimizer = optimizer
        self.loss = loss

        if model_builder:
            model_builder, model_builder_custom_options = import_model_builder(model_builder)
            self.model_builder = model_builder
            self.custom_objects.update(model_builder_custom_options)
            if 'name' not in self.model_builder_options:
                self.model_builder_options.update(name=self.name)
        self.model = None

    def predict(self, batch, batch_size=None):
        if self.model is None:
            self.__load_model()
        batch_size = batch_size if batch_size else self.batch_size
        prediction = self.model.predict(batch, batch_size=batch_size)
        return prediction

    def __load_model(self):
        from keras.models import load_model
        log.info("Loading keras model from %s", self.model_path)
        if not self.load_only_weights:
            self.model = load_model(self.model_path, custom_objects=self.custom_objects)
        else:
            self.model = self.__create_model()
        log.info("Finished loading")
        return self.model

    def __create_model(self):
        log.info("Building model")

        model_builder_options = self.model_builder_options
        if model_builder_options.get('input_shapes') is None:
            model_builder_options['input_shapes'] = self.input_shapes
        if model_builder_options.get('output_shapes') is None:
            model_builder_options['output_shapes'] = self.output_shapes
        model = self.model_builder(**model_builder_options)
        return model


    def fit_generator(self, train_data, validation_data=None):
        log.info("Training from generators")
        if self.steps_per_epoch is None:
            steps_per_epoch = len(train_data) // self.batch_size
        else:
            steps_per_epoch = self.steps_per_epoch

        if self.validation_steps_per_epoch is None:
            if validation_data is not None:
                validation_steps_per_epoch = len(validation_data) // self.batch_size
            else:
                validation_steps_per_epoch = None
        else:
            validation_steps_per_epoch = self.validation_steps_per_epoch

        if os.path.exists(self.model_path):
            log.info("Loading existing model")
            model = self.__load_model()
        else:
            model = self.__create_model()
            model.compile(self.optimizer, loss=self.loss)
            print (model.summary())



        model.fit_generator(train_data,
                            steps_per_epoch=steps_per_epoch,
                            epochs=self.epochs,
                            verbose=self.verbose,
                            callbacks=self.callbacks,
                            validation_data=validation_data,
                            validation_steps=validation_steps_per_epoch,
                            validation_freq=self.validation_freq,
                            class_weight=self.class_weight,
                            max_queue_size=self.max_queue_size,
                            workers=self.workers,
                            use_multiprocessing=self.use_multiprocessing,
                            shuffle=self.shuffle,
                            initial_epoch=self.initial_epoch)

def identity_metric(prediction, gti):
    return 1
