"""Lost and Found Object Detector — ranks candidate images by visual similarity
to a lost item using pretrained ResNet18 feature embeddings (no training needed),
explains the match in one sentence, and saves a Grad-CAM heatmap showing which
regions of the best candidate drove the match.
Usage: python detector.py [lost_image.jpg] — without an argument, the first
image found in the current folder is treated as the lost item."""
import glob
import os
import sys

# Force UTF-8 stdout so symbols like "—" print correctly in any console or pipe
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Load ResNet18 with pretrained ImageNet weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Replace the final FC layer with Identity -> model(x) now returns the
# 512-dim penultimate embedding instead of 1000 class scores
model.fc = torch.nn.Identity()
model.eval()  # inference mode (disables dropout/batchnorm updates)

# Standard ImageNet preprocessing: resize, convert to tensor, normalize
GRADCAM_OUT = "best_match_gradcam.jpg"  # heatmap output filename (skipped by the scan)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path):
    """Return the 512-dim feature vector for one image."""
    img = Image.open(image_path).convert("RGB")  # force 3 channels (jpg/png/grayscale)
    tensor = preprocess(img).unsqueeze(0)        # add batch dimension -> (1, 3, 224, 224)
    with torch.no_grad():                        # no gradients -> faster, less memory
        features = model(tensor)                 # shape (1, 512) since fc is Identity
    return features.squeeze().numpy()            # flatten to (512,)

def compare_images(lost_image_path, candidate_paths):
    """Rank candidates by cosine similarity to the lost image. Returns [(path, pct)]."""
    lost_vec = get_embedding(lost_image_path)
    sims = []
    for path in candidate_paths:
        cand_vec = get_embedding(path)
        # cosine similarity = dot(a, b) / (|a| * |b|); angle between feature vectors
        cos = np.dot(lost_vec, cand_vec) / (np.linalg.norm(lost_vec) * np.linalg.norm(cand_vec))
        sims.append(cos)
    # normalize raw similarities into percentages that sum to exactly 100
    total = sum(sims)
    percentages = [s / total * 100 for s in sims]
    # pair candidates with percentages, sort highest first
    return sorted(zip(candidate_paths, percentages), key=lambda x: x[1], reverse=True)

def describe_image(image_path):
    """Rough (color, shape) description used in the result sentence.
    Heuristic: dominant color = object pixels nearest a named palette color;
    shape = how much of the bounding box the object fills (square ~1.0,
    circle ~0.79, triangle ~0.5, star ~0.4). Works best on clean, single-object
    photos; the actual ranking is still done by the ResNet embeddings."""
    img = np.asarray(Image.open(image_path).convert("RGB").resize((64, 64)), dtype=float)
    # background = median color of the border pixels; object = pixels unlike it
    border = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), axis=0)
    mask = np.abs(img - border).sum(axis=2) > 90
    if not mask.any():
        mask = np.ones((64, 64), dtype=bool)
    ys, xs = np.where(mask)
    fill = mask.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    shape = ("boxy" if fill >= 0.9 else "round" if fill >= 0.65 else
             "triangular" if fill >= 0.42 else "star-like")
    palette = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (40, 70, 220),
               "yellow": (240, 220, 40), "orange": (250, 140, 30),
               "purple": (170, 50, 220), "pink": (250, 150, 170), "black": (30, 30, 30),
               "white": (240, 240, 240), "gray": (128, 128, 128), "brown": (140, 90, 50)}
    mean_rgb = img[mask].mean(axis=0)
    color = min(palette, key=lambda c: np.linalg.norm(mean_rgb - np.array(palette[c])))
    return color, shape

def make_gradcam(lost_image_path, best_image_path, out_path="best_match_gradcam.jpg"):
    """Grad-CAM heatmap for the best candidate. We backpropagate the cosine
    similarity score itself, so the heatmap highlights the regions of the
    candidate image that drove the match with the lost item. Target layer:
    layer4, ResNet18's last convolutional block (7x7 feature map)."""
    global GRADCAM_OUT
    GRADCAM_OUT = out_path
    acts, grads = {}, {}  # storage filled by the two hooks below
    def fwd(module, inp, out): acts["a"] = out            # capture activations
    def bwd(module, gin, gout): grads["g"] = gout[0]      # capture d(similarity)/d(acts)
    h1 = model.layer4.register_forward_hook(fwd)
    h2 = model.layer4.register_full_backward_hook(bwd)

    # forward pass WITH gradients enabled (unlike get_embedding) so we can backprop
    cand = preprocess(Image.open(best_image_path).convert("RGB")).unsqueeze(0)
    lost = torch.tensor(get_embedding(lost_image_path)).unsqueeze(0)  # fixed target
    sim = F.cosine_similarity(model(cand), lost)  # scalar similarity score
    sim.backward()                                 # gradients flow back into layer4
    h1.remove(); h2.remove()

    # Grad-CAM: weight each channel by its average gradient, sum, keep positives
    w = grads["g"].mean(dim=(2, 3), keepdim=True)           # (1, 512, 1, 1) weights
    cam = torch.relu((w * acts["a"]).sum(dim=1)).squeeze()  # (7, 7) importance map
    cam = cam / (cam.max() + 1e-8)                          # scale to [0, 1]

    # upscale to image size, colorize with a jet-style colormap, blend with original
    heat = Image.fromarray((cam.detach().numpy() * 255).astype("uint8")).resize((224, 224))
    anchors = [(0, 0, 143), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    t = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(t, [0, .25, .5, .75, 1], [a[i] for a in anchors])
                    for i in range(3)], axis=1)             # (256, 3) color lookup
    hm = lut[np.asarray(heat)]                              # (224, 224, 3) heatmap
    base = np.asarray(Image.open(best_image_path).convert("RGB").resize((224, 224)))
    blend = (0.5 * hm + 0.5 * base).clip(0, 255).astype("uint8")
    Image.fromarray(blend).save(out_path)
    return out_path

def main():
    # Auto-discover every .jpg/.png image in the current folder;
    # sort for a stable, predictable (alphabetical) order.
    # Skip our own Grad-CAM output so reruns don't treat it as a candidate.
    images = sorted(p for p in glob.glob("*.jpg") + glob.glob("*.png")
                    if p != GRADCAM_OUT)
    if not images:
        sys.exit("No .jpg/.png images found in this folder.")

    # Lost item: command-line argument if provided, else the first image found
    lost_image_path = sys.argv[1] if len(sys.argv) > 1 else images[0]
    if not os.path.isfile(lost_image_path):
        sys.exit(f"Lost image not found: {lost_image_path}")
    print(f"Lost item: {lost_image_path}")

    # All remaining images in the folder automatically become candidates
    lost_key = os.path.normcase(os.path.abspath(lost_image_path))
    candidate_paths = [p for p in images
                       if os.path.normcase(os.path.abspath(p)) != lost_key]
    if not candidate_paths:
        sys.exit("Need at least one candidate image besides the lost item.")

    ranked = compare_images(lost_image_path, candidate_paths)
    for path, pct in ranked:
        print(f"{path}: {pct:.1f}% match")
    best_path, best_pct = ranked[0]

    # Build the natural-language reason by comparing rough visual attributes
    lost_color, lost_shape = describe_image(lost_image_path)
    best_color, best_shape = describe_image(best_path)
    if best_color == lost_color and best_shape == lost_shape:
        reason = f"{best_color} color and {best_shape} form"
    elif best_color == lost_color:
        reason = f"its {best_color} color"
    elif best_shape == lost_shape:
        reason = f"its {best_shape} form"
    else:
        reason = "overall visual pattern"
    print(f"\nYour lost item is most likely {best_path}, with {best_pct:.1f}% "
          f"confidence, based on matching {reason}.")

    out = make_gradcam(lost_image_path, best_path)
    print(f"Grad-CAM heatmap for {best_path} saved to: {out}")

if __name__ == "__main__":
    main()
