import dtlpy as dl
import cv2
import logging
import numpy as np

logger = logging.getLogger("[Anonymizer]")


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def blur_objects(item: dl.Item, mask: np.array, sigma: int) -> np.array:
        logger.info("Blurring objects!")
        image = item.download(save_locally=False, to_array=True)

        # Convert the mask to the same size as the image
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_three_channels = np.stack([mask_resized] * 3, axis=-1)

        # Blur the objects in the image using Gaussian blur
        blurred_objects = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
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
                object_mask[int(object_of_interest.top):int(object_of_interest.bottom),
                            int(object_of_interest.left):int(object_of_interest.right)] = 1
            else:
                logger.warning("Object of interest is neither of type box nor mask.")
                object_mask = mask
            mask |= object_mask
        return mask

    @staticmethod
    def run_model(item: dl.Item, model: dl.Model, labels: list) -> dl.AnnotationCollection:
        logger.info("Starting model prediction")
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
        objects_of_interest = item.annotations.list(filters=interest_filter)
        logger.info(f"Number of objects of interest found in the image: {len(objects_of_interest)}")
        return objects_of_interest

    def anonymize_objects(self,
                          item: dl.Item,
                          model: dl.Model,
                          labels: list,
                          blur_intensity: int,
                          progress: dl.Progress) -> dl.Item:
        ## Run the model:
        objects_of_interest = self.run_model(item, model, labels)

        if len(objects_of_interest) > 0:
            mask = self.create_mask(item, objects_of_interest)
            blurred_image = self.blur_objects(item, mask, blur_intensity)
            blurred_item = item.dataset.items.upload(blurred_image,
                                                     remote_path="/.dataloop/blurred/",
                                                     remote_name=f"blurred_{model.name}_{item.name}")
            logger.info("Item for blurred image created!")
            blurred_item.metadata["original_item_id"] = item.id
            blurred_item = blurred_item.update()
            logger.info("Blurred item updated!")
            progress.update(action="anonymized")
        else:
            blurred_item = item
            logger.info("There were no objects of interest in the image")
            progress.update(action="no-objects")
        for ann in item.annotations.list():
            ann.delete()
        logger.info("Annotations deleted, original image cleaned up")
        return blurred_item
