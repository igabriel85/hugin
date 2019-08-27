import re
import os
import math

import h5py
import numpy as np
import rasterio

from logging import getLogger
from tempfile import TemporaryFile

from .core import NullMerger, postprocessor
from ..io import DataGenerator, DatasetGenerator
from ..io.loader import adapt_shape_and_stride


log = getLogger(__name__)


class MultipleScenePredictor:
    """
    This class is intended to be inherited by classes aimed to predict on multiple scenes
    """

    def __init__(self, scene_id_filter=None):
        """

        :param scene_id_filter: Regex for filtering scenes according to their id (optional)
        """

        self.scene_id_filter = None if not scene_id_filter else re.compile(scene_id_filter)

    def predict_scenes_proba(self, scenes, predictor=None):
        """Run the predictor on all input scenes

        :param scenes: An iterable object yielding tuples like (scene_id, type_mapping)
        :param predictor: The predictor to use for predicting scenes (defaults to self)
        :return: a list of predictions according to model configuration
        """

        predictor = self if predictor is None else predictor
        for scene_id, scene_data in scenes:
            if self.scene_id_filter and self.scene_id_filter.match(scene_id) is None:
                continue
            log.info("Classifying %s", scene_id)
            yield (scene_id, scene_data, predictor.predict_scene_proba((scene_id, scene_data)))

class BaseScenePredictor:
    def __init__(self, post_processors=None):
        self.post_processors = post_processors

    @postprocessor
    def predict_scene_proba(self, *args, **kwargs):
        raise NotImplementedError()

class CoreScenePredictor(BaseScenePredictor):
    def __init__(self, predictor,
                 name=None,
                 mapping=None,
                 stride_size=None,
                 output_shape=None,
                 prediction_merger=NullMerger,
                 post_processors=None):
        """

        :param predictor: Predictor to be used for predicting
        :param name: Name of the model (optional)
        :param mapping: Mapping of input files to input data
        :param stride_size: Stride size to be used by `predict_scene_proba`
        :param output_shape: Output shape of the prediction (Optional). Inferred from input image size
        """
        BaseScenePredictor.__init__(self, post_processors=post_processors)
        self.predictor = predictor

        instance_path = ".".join([self.__module__, self.__class__.__name__])
        self.name = "%s[%s]:%s" % (instance_path, name if name is not None else self.__hash__(), predictor.name)
        self.stride_size = stride_size if stride_size is not None else self.predictor.input_shape[0]
        if not mapping: raise TypeError("Missing `mapping` specification in %s" % self.name)
        self.mapping = mapping
        self.input_mapping = mapping.get('inputs')
        self.output_mapping = mapping.get('output', None)
        self.output_shape = output_shape
        self.prediction_merger_class = prediction_merger

    @postprocessor
    def predict_scene_proba(self, scene, dataset_loader=None):
        """Runs the predictor on the input scene
        This method might call `predict` for image patches

        :param scene: An input scene
        :return: a prediction according to model configuration
        """
        log.info("Generating prediction representation")
        scene_id, scene_data = scene
        output_mapping = self.mapping.get("target", {})
        tile_loader = DatasetGenerator((scene,)) if dataset_loader is None else dataset_loader.get_dataset_loader(scene)
        tile_loader.reset()
        data_generator = DataGenerator(tile_loader,
                                       batch_size=self.predictor.batch_size,
                                       input_mapping=self.input_mapping,
                                       output_mapping=None,
                                       swap_axes=self.predictor.swap_axes,
                                       loop=False,
                                       default_window_size=self.predictor.input_shape,
                                       default_stride_size=self.stride_size)

        if len(output_mapping) == 1:
            rio_raster = scene_data[output_mapping[0][0]]
            output_window_shape, output_stride_size = adapt_shape_and_stride(rio_raster,
                                                                             data_generator.primary_scene,
                                                                             self.predictor.input_shape,
                                                                             self.stride_size)
        else:
            output_window_shape = model_config.get('output_window_size', self.predictor.input_shape)
            output_stride_size = model_config.get('output_stride_size', self.stride_size)

        output_shape = self.output_shape
        if output_shape:
            output_height, output_width = output_shape
        else:
            output_height, output_width = data_generator.primary_scene.shape

        window_height, window_width = output_window_shape

        if window_width != self.stride_size:
            num_horizontal_tiles = math.ceil((output_width - window_width) / float(self.stride_size) + 2)
        else:
            num_horizontal_tiles = math.ceil((output_width - window_width) / float(self.stride_size) + 1)

        x_offset = 0
        y_offset = 0
        merger = None

        for data in data_generator:
            in_arrays, out_arrays = data
            prediction = self.predictor.predict(in_arrays, batch_size=self.predictor.batch_size)
            if merger is None:
                merger = self.prediction_merger_class(output_height, output_width, prediction.shape[3],
                                                      prediction.dtype)
            for i in range(0, in_arrays.shape[0]):
                tile_prediction = prediction[i].reshape(
                    (window_height, window_width, prediction.shape[3]))

                xstart = x_offset * self.stride_size
                xend = xstart + window_width
                ystart = y_offset * self.stride_size
                yend = ystart + window_height

                if xend > output_width:
                    xstart = output_width - window_width
                    xend = output_width
                if yend > output_height:
                    ystart = output_height - window_height
                    yend = output_height

                merger.update(xstart, xend, ystart, yend, tile_prediction)

                if x_offset == num_horizontal_tiles - 1:
                    x_offset = 0
                    y_offset += 1
                else:
                    x_offset += 1

        # image_probs = get_probabilities_from_tiles(
        #     self.predictor,
        #     data_generator,
        #     output_width,
        #     output_height,
        #     output_window_shape[0],
        #     output_window_shape[1],
        #     output_stride_size,
        #     self.predictor.batch_size,
        #     #merge_strategy=tile_merge_strategy
        # )
        prediction = merger.get_prediction()
        return prediction


class RasterScenePredictor(CoreScenePredictor, MultipleScenePredictor):
    def __init__(self, model, *args, scene_id_filter=None, **kwargs):
        CoreScenePredictor.__init__(self, model, *args, **kwargs)
        MultipleScenePredictor.__init__(self, scene_id_filter=scene_id_filter)


class BaseEnsembleScenePredictor(BaseScenePredictor, MultipleScenePredictor):
    def __init__(self, predictors,  *args, resume=False, cache_file=None, post_processors=None, **kwargs):
        BaseScenePredictor.__init__(self, post_processors=post_processors)
        MultipleScenePredictor.__init__(self, *args, **kwargs)
        self.predictors = predictors
        self.resume = resume
        cache_file = cache_file if cache_file is not None else TemporaryFile("w+b")
        self.cache = h5py.File(cache_file, 'a')
        log.info("Ensemble predictions stored in: %s", cache_file)

    def predict_scenes_proba(self, scenes):
        log.debug("Computing predictions for models")
        for predictor_configuration in self.predictors:
            scenes.reset()
            predictor = predictor_configuration["predictor"]
            log.info("Predicting using predictor: %s", predictor)
            for scene_id, _, prediction in super(BaseEnsembleScenePredictor, self).predict_scenes_proba(scenes, predictor):
                dataset_name = "%s/%s" % (predictor.name, scene_id)
                if self.resume and dataset_name in self.cache.keys():
                    log.info("Scene %s already predicted using %s. Skipping!", dataset_name, predictor.name)
                    continue
                log.debug("Storing prediction for %s under %s", scene_id, dataset_name)
                self.cache[dataset_name] = prediction

        scenes.reset()
        for scene in scenes:
            scene_id, scene_data = scene
            log.debug("Ensembling prediction for %s", scene_id)
            result = self.predict_scene_proba(scene)
            log.debug("Done ensembling")
            yield (scene_id, scene_data, result)


class AvgEnsembleScenePredictor(BaseEnsembleScenePredictor):
    @postprocessor
    def predict_scene_proba(self, scene):
        scene_id, scene_data = scene
        total_weight = 0
        sum_array = None
        for predictor_configuration in self.predictors:
            predictor = predictor_configuration["predictor"]
            predictor_weight = predictor_configuration.get("weight", 1)
            total_weight += predictor_weight
            dataset_name = "%s/%s" % (predictor.name, scene_id)
            log.debug("Using prediction from h5 dataset: %s", dataset_name)
            prediction = self.cache[dataset_name][()]
            if sum_array is None:
                sum_array = np.zeros(prediction.shape, dtype=prediction.dtype)
            sum_array += predictor_weight * prediction
        result = sum_array / total_weight
        return result


class SceneExporter(object):
    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination(self, destination):
        self._destination = destination

    def save_scene(self, scene_id, scene_data, prediction):
        raise NotImplementedError()

    def flow_from_source(self, loader, predictor):
        predictions = predictor.predict_scenes_proba(loader)
        for scene_id, scene_data, prediction in predictions:
            self.save_scene(scene_id, scene_data, prediction)


class RasterIOSceneExporter(SceneExporter):
    def __init__(self, destination,
                 srs_source_component=None,
                 rasterio_options={},
                 rasterio_creation_options={},
                 filename_pattern="{scene_id}.tif"):
        self.destination = destination
        self.srs_source_component = srs_source_component
        self.rasterio_options = rasterio_options
        self.rasterio_creation_options = rasterio_creation_options
        self.filename_pattern = filename_pattern

    def save_scene(self, scene_id, scene_data, prediction, destination_file=None):
        if destination_file is None:
            destination_file=self.filename_pattern.format(scene_id=scene_id)
        destination_file = os.path.join(self.destination, destination_file)
        log.info("Saving scene %s to %s", scene_id, destination_file)
        with rasterio.Env(**(self.rasterio_options)):
            profile = {}
            if self.srs_source_component is not None:
                src = scene_data[self.srs_source_component]
                profile.update (src.profile)
            num_out_channels = prediction.shape[-1]
            profile.update(self.rasterio_creation_options)
            profile.update(dtype=prediction.dtype, count=num_out_channels)
            if 'compress' not in profile:
                profile['compress'] = 'lzw'
            with rasterio.open(destination_file, "w", **profile) as dst:
                for idx in range(0, num_out_channels):
                    dst.write(prediction[:, :, idx], idx + 1)
