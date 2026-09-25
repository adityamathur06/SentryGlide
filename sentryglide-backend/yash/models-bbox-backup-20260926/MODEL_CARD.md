# SentryGlide medical-waste classifier

## Dataset

- Source archive: `C:\Users\yy207\Downloads\archive.zip`
- Imported RGB images: 1,415
- Classes: 13
- Original COCO train, validation, and test annotations were combined by category name.
- A fresh stratified 70/15/15 split was created after combining the source splits.
- Final split: 991 train, 212 validation, 212 test images.

The archive does not provide reliable specimen identities, so each source image is treated as a
separate physical ID. Similar consecutive photographs may still contain the same physical object;
the measured test score may therefore be optimistic.

## Model and results

- Input: 160 x 160 RGB plus ten image-derived geometric features
- Output: 13 object classes and a 96-value embedding
- Best validation accuracy: 88.68%
- Held-out test accuracy: 84.43%
- Confusion matrix: `confusion_matrix.png` and `confusion_matrix.csv`
- Per-class metrics: `classification_report.json`

The most important remaining confusions are between visually similar glove material/style classes
and between single and paired shoe covers. At the disposal-bin level, those subclasses map to the
same RED bin.

## Routing limitations

- `test_tube` is mapped to HOLD because the dataset does not identify whether a tube is glass or plastic.
- This dataset contains no sharps/WHITE-bin examples and no BLUE-bin glass/metal classes.
- It is not sufficient for complete four-bin biomedical-waste deployment.
- The formal 0.1% dangerous-route upper-bound requirement is not met because the held-out sample is
  too small. Approximately 3,000 independent validation placements with zero dangerous errors are
  required for that statistical target.

This model is for local prototype demonstration and research evaluation only.

