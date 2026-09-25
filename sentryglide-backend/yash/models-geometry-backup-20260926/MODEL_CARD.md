# SentryGlide medical-waste classifier

## Dataset and preprocessing

- Source: 1,415 RGB images from the supplied COCO medical-waste archive.
- Classes: 13.
- Split: 991 train, 212 validation, 212 held-out test images.
- The original archive splits were combined, then a fresh 70/15/15 split was created.
- Training and upload inference now use the same foreground detection, crop, letterbox, mask,
  and geometric-feature pipeline.

The archive does not identify repeated physical specimens, so visually related photographs may
cross the split boundary. Reported accuracy may therefore be optimistic.

## Results

- Best validation accuracy: 91.51%
- Held-out test accuracy: 83.49%
- Upload-path check on 120 archive glove photographs: 88.33% subtype accuracy
- Before preprocessing alignment, the same upload-path check achieved only 61.67%.

Most remaining errors are between pair/single or material subtypes of gloves. Those subclasses all
map to RED, so these subtype errors do not change the disposal-bin recommendation.

## Limitations

- `test_tube` is held because the dataset does not specify glass versus plastic.
- The dataset contains no sharps/WHITE examples and no BLUE-bin glass/metal examples.
- This model is for prototype demonstration, not autonomous biomedical-waste routing.
