# Anonymizer

This is a pipeline node that takes an image item, an object detection or segemnetation model, and a list of labels.
The node will run a prediction with the chosen model (which must be already deployed as a different service), check
whether there were detections of the input labels and, if that is the case, blur those objects. All the annotations are
then deleted.