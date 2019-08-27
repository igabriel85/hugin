from logging import getLogger

import numpy as np

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
                 input_shape=None,
                 output_shape=None):
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
        self.input_shape = input_shape
        self.output_shape = output_shape

    def predict(self, batch):
        """Runs the predictor on the input tile batch

        :param batch: The input batch
        :return: returns a prediction according to model configuration
        """
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
    def __init__(self, model_path, model_builder, *args, custom_objects={}, **kwargs):
        RasterModel.__init__(self, *args, **kwargs)
        self.custom_objects = custom_objects
        self.model_path = model_path
        if model_builder:
            _, model_builder_custom_options = import_model_builder(model_builder)
            custom_objects.update(model_builder_custom_options)
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
        self.model = load_model(self.model_path, custom_objects=self.custom_objects)
        log.info("Finished loading")
        return self.model

def identity_metric(prediction, gti):
    return 1
