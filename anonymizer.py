import dtlpy as dl
from typing import Any
import cv2
import logging
import numpy as np
import time
from typing import List

logger = logging.getLogger("[Anonymizer]")


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def blur_objects(item: dl.Item, mask: np.array, sigma: int, blur: bool = True) -> np.array:
        logger.info("Blurring objects!")
        download_time_start = time.time()
        image = item.download(save_locally=False, to_array=True)
        download_end_time = time.time()
        print(f"Downloading time spent: {download_end_time - download_time_start}")

        three_channels_start_time = time.time()
        image_three_channels = image if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
        print(f"Creating three channels image time spent: {time.time() - three_channels_start_time}")

        # Convert the mask to the same size as the image
        mask_three_channels_start = time.time()
        mask_three_channels = np.stack([mask] * 3, axis=-1)
        print(f"Creating three channels mask time spent: {time.time() - mask_three_channels_start}")

        if blur is True:
            # Blur the objects in the image using Gaussian blur
            blur_start = time.time()
            blurred_objects = cv2.GaussianBlur(image_three_channels,
                                               (0, 0),
                                               sigmaX=sigma,
                                               sigmaY=sigma,
                                               borderType=cv2.BORDER_DEFAULT)
            print(f"Blurring time spent: {time.time() - blur_start}")
        else:
            blurred_objects = mask_three_channels
        logger.info("Blurred version created!")
        replace_start = time.time()
        result = np.where(mask_three_channels, blurred_objects, image_three_channels)
        print(f"Object replacing time spent: {time.time() - replace_start}")
        logger.info("Objects blurred.")
        return result

    @staticmethod
    def create_mask(item: dl.Item, objects_of_interest: dl.AnnotationCollection):
        mask = np.zeros((item.height, item.width), dtype=np.uint8)
        for i, object_of_interest in enumerate(objects_of_interest):
            logger.info(f"Mask for object {i} being created")
            object_mask = np.zeros_like(mask, dtype=np.uint8)
            if object_of_interest.type == dl.ANNOTATION_TYPE_POLYGON and len(object_of_interest.geo) > 0:
                # Generate a polygon mask
                segmentation = dl.Segmentation.from_polygon(object_of_interest.geo,
                                                            object_of_interest.label,
                                                            (item.height, item.width))
                object_mask = np.array(segmentation.geo, dtype=np.uint8)
            elif object_of_interest.type == dl.ANNOTATION_TYPE_BOX:
                # Generate a box mask
                top, bottom = int(object_of_interest.top), int(object_of_interest.bottom)
                left, right = int(object_of_interest.left), int(object_of_interest.right)
                if 0 <= top <= bottom <= item.height and 0 <= left <= right <= item.width:
                    object_mask[top:bottom, left:right] = 1
                else:
                    raise ValueError(f"Detection {object_of_interest.id} has coordinates outside of the image!")
            elif object_of_interest.type == dl.ANNOTATION_TYPE_SEGMENTATION:
                # Use the segmentation mask
                object_mask = np.array(object_of_interest.geo, dtype=np.uint8)
            else:
                logger.warning("Object of interest is neither of type box nor mask.")
                object_mask = mask
            mask |= object_mask
        return mask

    @staticmethod
    def run_model(item: dl.Item, model: dl.Model, labels: list) -> dl.AnnotationCollection:
        logger.info("Starting model prediction")
        if model.status != "deployed" or \
                len(model.metadata.get("system", {}).get("deploy", {}).get("services", [])) == 0:
            raise Exception(f"Model {model.id} is not deployed! Can't run anonymization.")
        predict_execution = model.predict([item.id])
        logger.info("Waiting for prediction results")
        predict_execution.wait()
        logger.info("Prediction ended successfully.")
        interest_filter = dl.Filters(
            dl.KnownFields.LABEL,
            labels,
            operator=dl.FiltersOperations.IN,
            resource=dl.FiltersResource.ANNOTATION
        )
        interest_filter.add("metadata.system.model.model_id", model.id)
        objects_of_interest = item.annotations.list(filters=interest_filter)
        logger.info(f"Number of objects of interest found in the image: {len(objects_of_interest)}")
        return objects_of_interest

    @staticmethod
    def get_models_and_labels(context: dl.Context) -> tuple[Any, Any, Any]:
        node = context.node
        model_ids = node.metadata['customNodeConfig'].get('model_ids', "")
        model_ids = model_ids.split(",")
        models_list = [dl.models.get(model_id=model_id) if model_id != "" else None for model_id in model_ids]
        labels = node.metadata['customNodeConfig']['labels']
        labels = labels.split(",")
        logger.info("Model loaded")
        logger.info(f"Model to be used: {model_ids}. Labels to be searched: {labels}")
        return models_list, model_ids, labels

    def predict_and_anonymize(self, item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
        logger.info("Starting prediction+anonymization")
        models_list, model_ids, labels = self.get_models_and_labels(context)
        res = None
        for model in models_list:
            logger.info(f"Running prediction for model {model.id}.")
            model_run_start = time.time()
            objects_of_interest = self.run_model(item, model, labels)
            print(f"----- Time spent in model prediction: {time.time() - model_run_start}")
            logger.info(f"Prediction successful")
            anon_start = time.time()

            res = self.anonymize_objects(item, objects_of_interest, model.id, progress, context)
            print(f"----- Anonymization time: {time.time() - anon_start}")
        return res

    def anonymize(self, item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
        logger.info("Starting anonymization w/o prediction")
        models_list, model_ids, labels = self.get_models_and_labels(context)
        res = None
        for model in models_list:
            objects_of_interest_filter = dl.Filters(dl.KnownFields.LABEL,
                                                    labels,
                                                    resource=dl.FiltersResource.ANNOTATION,
                                                    operator=dl.FiltersOperations.IN)
            objects_of_interest_filter.add("metadata.system.model.model_id", model.id)
            logger.info(f"Filtering annotations generated from the model {model.id}.")
            objects_of_interest = item.annotations.list(filters=objects_of_interest_filter)
            anon_start = time.time()
            res = self.anonymize_objects(item, objects_of_interest, model.id, progress, context)
            print(f"&&&&&& Anonymization time: {time.time() - anon_start}")
        return res

    def anonymize_annotations(self,
                              annotations: dl.AnnotationCollection,
                              progress: dl.Progress,
                              context: dl.Context) -> dl.Item:
        logger.info("Starting anonymization based on annotations.")
        _, _, labels = self.get_models_and_labels(context)
        item = annotations[0].item
        filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        filters.add(field='label', values=labels, operator=dl.FILTERS_OPERATIONS_IN)
        objects_of_interest = item.annotations.list(filters=filters)
        anon_start = time.time()
        res = self.anonymize_objects(item, objects_of_interest, "", progress, context)
        print(f"&&&&&& Anonymization time: {time.time() - anon_start}")
        return res

    def anonymize_objects(self,
                          item: dl.Item,
                          objects_of_interest: dl.AnnotationCollection,
                          model_id: str,
                          progress: dl.Progress,
                          context: dl.Context
                          ) -> dl.Item:
        # Initialization
        logger.info("Initializing the parameters from the node configuration")
        node = context.node
        blur_intensity = node.metadata['customNodeConfig']['blur_intensity']
        blur = node.metadata['customNodeConfig'].get('blur')
        blur = "blur" in blur if blur else True
        replace = node.metadata['customNodeConfig'].get('replace')
        replace = "yes" in replace if replace else True
        dataset_id = item.dataset_id
        remote_path = node.metadata["customNodeConfig"].get("directory", "/blurred")
        labels = node.metadata['customNodeConfig']['labels']
        prefix = "blurred"

        logger.debug(f"INPUT CONFIGURATIONS FOUND -- blur_intensity: {blur_intensity}, dataset_id: {dataset_id}, "
                     f"remote_path: {remote_path}, prefix: {prefix}, blur: {blur}, replace: {replace}")

        dataset = dl.datasets.get(dataset_id=dataset_id)
        logger.info("Dataset loaded")

        logger.info(f"Obtained {len(objects_of_interest)} objects of interest.")

        if len(objects_of_interest) > 0:
            mask_creation_start = time.time()
            mask = self.create_mask(item, objects_of_interest)
            print(f"*** Time spent in mask creation: {time.time() - mask_creation_start}")
            logger.info("Mask created.")
            blur_start = time.time()
            blurred_image = self.blur_objects(item, mask, blur_intensity, blur)
            print(f"*** Time spent in blurring: {time.time() - blur_start}")
            logger.info("Blurred image created")
            metadata = item.metadata
            metadata["anonymization"] = {"original_item_id": item.id, "anonymized": True}
            blurred_item = dataset.items.upload(blurred_image,
                                                remote_path=remote_path,
                                                item_metadata=metadata,
                                                remote_name=f"{prefix}_{item.name}")
            blurred_item.annotations.upload(objects_of_interest)
            logger.info("Item for blurred image created!")
            logger.info("Blurred item updated!")
            if replace == "replace":
                logger.info("Replacing original item with blurred item.")
                item.modalities.delete(name="reference-viewer")
                item.modalities.create(modality_type=dl.MODALITY_TYPE_REPLACE,
                                       name='reference-viewer',
                                       mimetype=blurred_item.mimetype,
                                       ref=blurred_item.id
                                       )
            elif replace == "remove":
                logger.info("Removing original item.")
                item.delete()
            else:
                logger.info("Original item was kept unchanged.")
            if 'user' not in item.metadata:
                item.metadata['user'] = dict()
            item.metadata['user']['result_item_id'] = blurred_item.id
            item.update(system_metadata=True)
            progress.update(action="anonymized")
        else:
            blurred_item = item
            logger.info("There were no objects of interest in the image")
            blurred_item.metadata["anonymization"] = {"anonymized": False}
            progress.update(action="no-objects")
        annotation_deletion_filter = dl.Filters(resource=dl.FILTERS_RESOURCE_ANNOTATION)
        for label in labels:
            annotation_deletion_filter.add(dl.KnownFields.LABEL, label, operator=dl.FiltersOperations.NOT_EQUAL)
        if model_id != "":
            annotation_deletion_filter.add("metadata.system.model.model_id", model_id)
        item.annotations.delete(filters=annotation_deletion_filter)
        blurred_item = blurred_item.update(system_metadata=True)
        logger.info("Annotations deleted, original image cleaned up")
        return blurred_item
