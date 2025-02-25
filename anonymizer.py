import dtlpy as dl
import cv2
import os
import logging
import numpy as np

logger = logging.getLogger("[Anonymizer]")


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def blur_objects(item: dl.Item, mask: np.array, sigma: int, blur: bool = True) -> np.array:
        logger.info("Blurring objects!")

        image = item.download(save_locally=False, to_array=True)
        image_three_channels = image if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)

        # Convert the mask to the same size as the image
        mask_three_channels = np.stack([mask] * 3, axis=-1)

        if blur is True:
            # Blur the objects in the image using Gaussian blur
            blurred_objects = cv2.GaussianBlur(
                image_three_channels, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT
            )
        else:
            blurred_objects = mask_three_channels
        logger.info("Blurred version created!")
        result = np.where(mask_three_channels, blurred_objects, image_three_channels)
        logger.info("Objects blurred.")
        return result

    @staticmethod
    def create_mask(item: dl.Item, objects_of_interest: dl.AnnotationCollection):
        mask = np.zeros((item.height, item.width), dtype=np.uint8)
        for i, object_of_interest in enumerate(objects_of_interest):
            object_of_interest: dl.Annotation
            logger.info(f"Mask for object {i} being created")
            object_mask = np.zeros_like(mask, dtype=np.uint8)
            if object_of_interest.type == dl.ANNOTATION_TYPE_POLYGON and len(object_of_interest.geo) > 0:
                # Generate a polygon mask
                segmentation = dl.Segmentation.from_polygon(
                    object_of_interest.geo, object_of_interest.label, (item.height, item.width)
                )
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
    def get_models_and_labels(context: dl.Context) -> tuple[list[str], list[str]]:
        """
        Retrieve model IDs and labels from the node's custom configuration.

        Args:
            context (dl.Context): Context containing node metadata.

        Returns:
            tuple[list[str], list[str]]: Lists of model IDs and labels.
        """
        node = context.node
        model_ids = node.metadata["customNodeConfig"].get("model_ids", "")
        labels = node.metadata["customNodeConfig"].get("labels", "")

        # Process model IDs and labels into lists, ignoring empty values
        model_ids = [model_id.strip() for model_id in model_ids.split(",") if not model_id.strip() == ""]
        labels = [label.strip() for label in labels.split(",") if not label.strip() == ""]

        logger.info("Configuration retrieved:")
        logger.info(f"Model IDs: {model_ids if model_ids else 'None'}")
        logger.info(f"Labels: {labels if labels else 'None'}")

        return model_ids, labels

    def anonymize(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Anonymize annotations for a given item based on models and labels.

        Args:
            item (dl.Item): The item to process.
            context (dl.Context): Context for accessing node configuration.

        Returns:
            dl.Item: The updated item.
        """
        logger.info("Starting anonymization process.")
        model_ids, labels = self.get_models_and_labels(context)

        # Prepare the base filter for annotations
        filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
        if len(labels) != 0:  # Add label filter if labels are provided
            filters.add(dl.KnownFields.LABEL, labels, operator=dl.FiltersOperations.IN)
            logger.info(f"Filtering annotations for labels: {labels}")

        # If models are specified, process for each model
        if len(model_ids) != 0:
            filters.add(field="metadata.system.model.model_id", values=model_ids, operator=dl.FiltersOperations.IN)
            logger.info(f"Filtering annotations for model IDs: {model_ids}")

        objects_of_interest = item.annotations.list(filters=filters)
        item = self.anonymize_objects(item, objects_of_interest, context)

        logger.info("Anonymization process completed.")
        return item

    def anonymize_objects(
        self, item: dl.Item, objects_of_interest: dl.AnnotationCollection, context: dl.Context
    ) -> dl.Item:
        """
        Anonymizes objects in the provided item based on configuration, while preserving other annotations.

        Args:
            item (dl.Item): The Dataloop item to be anonymized.
            objects_of_interest (dl.AnnotationCollection): Annotations to be anonymized.
            context (dl.Context): Dataloop context for node configuration.

        Returns:
            dl.Item: The anonymized item.
        """
        # Retrieve configuration
        logger.info("Initializing parameters from the node configuration.")
        node_config = context.node.metadata["customNodeConfig"]
        blur_intensity = node_config["blur_intensity"]
        blur = node_config.get("blur", "").lower() == "blur"
        anonymization_type = node_config.get("anonymization_type")
        dataset = item.dataset
        remote_path = node_config.get("directory", "/blurred")
        prefix = "blurred"

        logger.debug(
            f"INPUT CONFIGURATIONS -- blur_intensity: {blur_intensity}, dataset_id: {dataset.id}, "
            f"remote_path: {remote_path}, prefix: {prefix}, blur: {blur}, anonymization_type: {anonymization_type}"
        )

        logger.info(f"Found {len(objects_of_interest)} objects of interest.")

        # Handle objects of interest
        if len(objects_of_interest) > 0:
            # Define the filter to exclude annotations that are in objects_of_interest
            filters_not_object_of_interest = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
            filters_not_object_of_interest.add(
                field="id", values=[ann.id for ann in objects_of_interest], operator=dl.FiltersOperations.NIN
            )

            # Apply the filter to retrieve annotations that do not match the objects of interest
            other_annotations = item.annotations.list(filters=filters_not_object_of_interest)

            # Create mask and apply blur
            mask = self.create_mask(item, objects_of_interest)
            logger.info("Mask created successfully.")
            blurred_image = self.blur_objects(item, mask, blur_intensity, blur)
            logger.info("Blurred image created.")

            # Handle metadata for the blurred item
            original_item_metadata = item.metadata.copy()
            blurred_item = None

            if anonymization_type == "replace":
                logger.info("Anonymization type: replace. Overwriting original item with blurred item.")
                blurred_item = dataset.items.upload(
                    blurred_image,
                    remote_path=item.dir,
                    remote_name=item.name,
                    item_metadata=original_item_metadata,
                    overwrite=True,
                )
                # Upload both objects of interest and other annotations
                blurred_item.annotations.upload(objects_of_interest)
                blurred_item.annotations.upload(other_annotations)
                logger.info("Original item replaced successfully.")
            elif anonymization_type == "remove":
                logger.info("Anonymization type: remove. Creating a new item and removing the original.")
                blurred_item = dataset.items.upload(
                    blurred_image,
                    remote_path=remote_path,
                    remote_name=f"{prefix}_{item.name}",
                    item_metadata=original_item_metadata,
                )
                # Upload both objects of interest and other annotations
                blurred_item.annotations.upload(objects_of_interest)
                blurred_item.annotations.upload(other_annotations)
                item.delete()
                logger.info("Original item removed successfully.")
            else:
                logger.info("Anonymization type: keep. Creating a new blurred item and keeping the original.")
                original_item_metadata["user"] = original_item_metadata.get("user", {})
                original_item_metadata["user"]["original_item_id"] = item.id

                blurred_item = dataset.items.upload(
                    blurred_image,
                    remote_path=remote_path,
                    remote_name=f"{prefix}_{item.name}",
                    item_metadata=original_item_metadata,
                    overwrite=True,
                )
                # Upload both objects of interest and other annotations
                blurred_item.annotations.upload(objects_of_interest)
                blurred_item.annotations.upload(other_annotations)

                item.metadata["user"] = item.metadata.get("user", {})
                item.metadata["user"]["anonymization"] = {"anonymized": True}
                item.metadata["user"]["blurred_item_id"] = blurred_item.id
                item.update()
                logger.info("Original item retained and linked to blurred item.")
        else:
            # No objects to anonymize
            logger.info("No annotations to anonymize in the image based on labels provided.")
            item.metadata["user"] = item.metadata.get("user", {})
            item.metadata["user"]["anonymization"] = {"anonymized": False}
            item.update()
            blurred_item = item.clone(
                dst_dataset_id=item.dataset_id,
                remote_filepath=remote_path + "/" + item.name,
                with_annotations=True,
                with_metadata=True,
            )
        blurred_item.update()
        logger.info("Blurred item metadata updated.")
        return blurred_item


if __name__ == "__main__":
    runner = ServiceRunner()
    context = dl.Context()
    context.pipeline_id = ""
    context.node_id = ""
    context.node.metadata["customNodeConfig"] = {
        "blur_intensity": 10,
        "blur": "blur",
        "anonymization_type": "keep",
        "directory": "/testing",
    }
    runner.anonymize(item=dl.items.get(item_id=""), context=context)
