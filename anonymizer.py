import dtlpy as dl
import cv2
import logging
import numpy as np

logger = logging.getLogger("[Anonymizer]")


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def blur_objects(item: dl.Item, mask: np.array, sigma: int, blur: bool = True) -> np.array:
        logger.info("Blurring objects!")
        image = item.download(save_locally=False, to_array=True)

        # Convert the mask to the same size as the image
        mask_three_channels = np.stack([mask] * 3, axis=-1)

        if blur:
            # Blur the objects in the image using Gaussian blur
            blurred_objects = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
        else:
            blurred_objects = mask_three_channels
        logger.info("Blurred version created!")
        result = np.where(mask_three_channels, blurred_objects, image)
        logger.info("Objects blurred.")
        return result

    @staticmethod
    def create_mask(item: dl.Item, objects_of_interest: dl.AnnotationCollection):
        mask = np.zeros((item.height, item.width), dtype=np.uint8)
        for i, object_of_interest in enumerate(objects_of_interest):
            logger.info(f"Mask for object {i} being created")
            if object_of_interest.type == dl.ANNOTATION_TYPE_POLYGON:
                object_mask = dl.Segmentation.from_polygon(object_of_interest.geo,
                                                           object_of_interest.label,
                                                           (item.height, item.width)).geo
            elif object_of_interest.type == dl.ANNOTATION_TYPE_BOX:
                object_mask = np.zeros((item.height, item.width))
                top, bottom = int(object_of_interest.top), int(objects_of_interest.bottom)
                left, right = int(object_of_interest.left), int(objects_of_interest.right)
                if 0 <= top <= bottom <= item.height and 0 <= left <= right <= item.width:
                    object_mask[top:bottom, left:right] = 1
                else:
                    raise Exception(f"Detection {object_of_interest.id} has coordinates outside of the image!")
            elif object_of_interest.type == dl.ANNOTATION_TYPE_SEGMENTATION:
                object_mask = object_of_interest.geo
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
            "label",
            labels,
            operator=dl.FiltersOperations.IN,
            resource=dl.FiltersResource.ANNOTATION
            )
        interest_filter.add("metadata.system.model.model_id", model.id)
        objects_of_interest = item.annotations.list(filters=interest_filter)
        logger.info(f"Number of objects of interest found in the image: {len(objects_of_interest)}")
        return objects_of_interest

    def anonymize_objects(self,
                          item: dl.Item,
                          progress: dl.Progress,
                          context: dl.Context
                          ) -> dl.Item:
        # Initialization
        logger.info("Initializing the parameters from the node configuration")
        node = context.node
        model_id = node.metadata['customNodeConfig']['model_id']
        blur_intensity = node.metadata['customNodeConfig']['blur_intensity']
        labels = node.metadata['customNodeConfig']['labels']
        dataset_id = node.metadata['customNodeConfig'].get('dataset_id')
        dataset_id = dataset_id if dataset_id else item.dataset_id
        remote_path = node.metadata['customNodeConfig'].get('remote_path')
        remote_path = remote_path if remote_path else "/.dataloop/blurred"
        prefix = node.metadata['customNodeConfig'].get('prefix')
        prefix = prefix if prefix else "blurred"
        blur = node.metadata['customNodeConfig'].get('blur')
        blur = "blur" in blur if blur else True
        replace = node.metadata['customNodeConfig'].get('replace')
        replace = "yes" in replace if replace else True
        logger.debug(f"INPUT CONFIGURATIONS FOUND -- model_id: {model_id}, blur_intensity: {blur_intensity}, labels: "
                     f"{labels}, dataset_id: {dataset_id}, remote_path: {remote_path}, prefix: {prefix}, blur: {blur} "
                     f"replace: {replace}")

        labels = labels.split(",")
        dataset = dl.datasets.get(dataset_id=dataset_id)
        logger.info("Dataset loaded")
        model = dl.models.get(model_id=model_id)
        logger.info("Model loaded")
        # Run the model:
        logger.info("Running model to obtain detections")
        objects_of_interest = self.run_model(item, model, labels)
        logger.info(f"Model run successful. Obtained {len(objects_of_interest)} objects of interest.")

        if len(objects_of_interest) > 0:
            mask = self.create_mask(item, objects_of_interest)
            logger.info("Mask created.")
            blurred_image = self.blur_objects(item, mask, blur_intensity, blur)
            logger.info("Blurred image created")
            blurred_item = dataset.items.upload(blurred_image,
                                                remote_path=remote_path,
                                                remote_name=f"{prefix}_{item.name}")
            logger.info("Item for blurred image created!")
            blurred_item.metadata["original_item_id"] = item.id
            blurred_item.metadata["anonymized"] = True
            blurred_item = blurred_item.update()
            logger.info("Blurred item updated!")
            if replace:
                logger.info("Replacing original item with blurred item.")
                item.modalities.delete(name="reference-viewer")
                item.modalities.create(modality_type=dl.MODALITY_TYPE_OVERLAY,
                                       name='reference-viewer',
                                       mimetype=blurred_item.mimetype,
                                       ref=blurred_item.id
                                       )
                item.update(system_metadata=True)
            progress.update(action="anonymized")
        else:
            blurred_item = item
            logger.info("There were no objects of interest in the image")
            blurred_item.metadata["system"]["anonymized"] = False
            progress.update(action="no-objects")
        for ann in item.annotations.list():
            ann.delete()
        blurred_item = blurred_item.update(system_metadata=True)
        logger.info("Annotations deleted, original image cleaned up")
        return blurred_item
