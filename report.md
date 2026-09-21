# Lost and Found Object Detector — Project Report

## 1. Problem Statement

When a personal item is lost, the owner is often presented with a small set of candidate objects recovered or found elsewhere, and must decide which (if any) is theirs. Manually comparing photos of the lost item against candidates is error-prone when items are visually similar. This project implements an automated, training-free pipeline that ranks candidate images by visual similarity to a photo of the lost item and identifies the most likely match with a normalized confidence score.

## 2. Methodology

The system uses ResNet18 pretrained on ImageNet as a fixed feature extractor. The final fully connected layer is replaced with an identity operation, so a forward pass yields the 512-dimensional penultimate embedding rather than class predictions. Each image is converted to RGB, resized to 224×224, and normalized with the ImageNet channel means and standard deviations to match the network's training distribution. Cosine similarity is computed between the lost item's embedding and each candidate's embedding, which measures the angle between feature vectors and is therefore insensitive to vector magnitude. The raw similarity scores are normalized to sum to 100%, producing an interpretable relative distribution, and candidates are sorted in descending order. A lightweight heuristic (dominant color via palette distance; shape via bounding-box fill ratio) generates a short natural-language justification for the best match. The pipeline runs entirely on CPU and requires no training or fine-tuning.

## 3. Generalization to Arbitrary Object Categories

The approach does not depend on the specific object categories used in development, because the underlying features were learned on ImageNet's 1.2 million images spanning 1,000 diverse classes. During training, ResNet18's convolutional layers learned a hierarchy of increasingly abstract features — edges, textures, object parts, and whole-object layouts — rather than detectors for fixed categories. Consequently, the 512-dimensional embedding provides a useful vector representation for object types never seen as labeled classes, including water bottles, keys, bags, or electronics. Only the final classification step was category-bound, and it is discarded; the representation itself is general-purpose.

## 4. Grad-CAM Explainability

For the best-matching candidate, the system generates a Grad-CAM heatmap targeting `layer4` (the final convolutional block). Unlike the standard formulation, which backpropagates a class score, this implementation backpropagates the cosine similarity between the candidate and the lost item. The resulting heatmap therefore highlights the image regions that most influenced the *match decision* rather than a category prediction. The 7×7 importance map is upsampleled, colorized with a jet-style colormap, and blended with the original image at 50% opacity. This matters for two reasons: it provides evidence for the report that similarity is driven by the object itself rather than background context, and it offers end users a sanity check — if the heatmap concentrates on a chair behind the item rather than the item, the match can be distrusted.

## 5. Limitations

- **Mass-produced items.** Cosine similarity in a general feature space may not distinguish two nearly identical objects (e.g., same-model phone chargers); embeddings capture object *type*, not instance-level identity such as serial numbers or scratches.
- **Relative, not absolute, scores.** Because scores are normalized to sum to 100%, a weak best match can still display high confidence when all candidates differ greatly; the percentage expresses relative ranking, not probability of a true match.
- **Viewpoint and context sensitivity.** Extreme viewpoints, occlusion, cluttered backgrounds, or poor lighting can shift embeddings enough to change the ranking.
- **Heuristic justification.** The color/shape reason in the output sentence is computed by a simple pixel heuristic, not by the network, and may misdescribe complex photos; the ranking itself is unaffected.
- **No rejection mechanism.** The system always names a best match even if the lost item is absent from the candidate pool.
