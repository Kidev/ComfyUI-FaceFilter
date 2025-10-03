import os
import numpy as np
import torch
import cv2
from insightface.app import FaceAnalysis

_FACE_APP_CACHE = {}

def _packs_under(root):
    out = []
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and any(f.endswith(".onnx") for f in os.listdir(p)):
                out.append(name)
    except Exception:
        pass
    return out

def _default_models_root():
    local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "insightface"))
    home_root = os.path.expanduser("~/.insightface/models")
    return local_root if os.path.isdir(local_root) else home_root

def _existing_pack_names():
    candidates = ("antelopev2", "buffalo_l", "buffalo_m", "buffalo_s")
    roots = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "insightface")),
        os.path.expanduser("~/.insightface/models"),
    )
    found = []
    seen = set()
    for name in candidates:
        for root in roots:
            p = os.path.join(root, name)
            if os.path.isdir(p) and any(f.lower().endswith(".onnx") for f in os.listdir(p)):
                if name not in seen:
                    found.append(name)
                    seen.add(name)
                break
    if not found:
        return ("antelopev2",)
    return tuple(found)

_PACK_CHOICES = _existing_pack_names()

def _providers_from_choice(choice):
    if choice == "cpu_only":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

def _get_face_app(model_name, providers_choice, det_size):
    model_root = _default_models_root()
    providers = _providers_from_choice(providers_choice)
    key = (os.path.abspath(model_root), model_name, tuple(providers), int(det_size))
    app = _FACE_APP_CACHE.get(key)
    if app is None:
        app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        app.prepare(ctx_id=0, det_size=(int(det_size), int(det_size)))
        _FACE_APP_CACHE[key] = app
    return app

def _ensure_uint8_rgb(t):
    if isinstance(t, torch.Tensor):
        x = t.detach().cpu()
        if x.ndim == 3 and x.shape[-1] in (3, 4):
            rgb = x[..., :3].numpy()
        elif x.ndim == 3 and x.shape[0] in (3, 4):
            rgb = x[:3, ...].permute(1, 2, 0).numpy()
        else:
            raise ValueError("Unsupported tensor shape for image")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb
    if isinstance(t, np.ndarray):
        arr = t
        if arr.ndim != 3:
            raise ValueError("Unsupported numpy image shape")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr
    raise ValueError("Unsupported image type")

def _prepare_for_detection(img_tensor_or_np, size, mode):
    rgb = _ensure_uint8_rgb(img_tensor_or_np)
    h, w, _ = rgb.shape
    scale = (float(size) / float(min(h, w))) if mode == "fit_min_side" else (float(size) / float(max(h, w)))
    if scale != 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    return bgr, (h, w)

def _iterate_batch(imgs):
    if isinstance(imgs, torch.Tensor):
        if imgs.ndim == 4:
            for i in range(imgs.shape[0]):
                yield i, imgs[i]
        elif imgs.ndim == 3:
            yield 0, imgs
        else:
            raise ValueError("Expected IMAGE tensor of ndim 3 or 4")
    elif isinstance(imgs, (list, tuple)):
        for i, im in enumerate(imgs):
            yield i, im
    else:
        yield 0, imgs

def _as_batched_tensor(imgs):
    if isinstance(imgs, torch.Tensor):
        return imgs if imgs.ndim == 4 else imgs.unsqueeze(0)
    if isinstance(imgs, (list, tuple)):
        ts = []
        for im in imgs:
            if not isinstance(im, torch.Tensor):
                raise ValueError("IMAGE list must contain torch tensors")
            ts.append(im if im.ndim == 3 else im.squeeze(0))
        return torch.stack(ts, dim=0) if ts else torch.zeros((0, 64, 64, 3), dtype=torch.float32)
    raise ValueError("Unsupported IMAGE container")

def _empty_like_batch(batched):
    return batched[0:0]

def _pass_through_first(batched):
    return batched[:1] if batched.shape[0] else batched

def _black_image_batch_like(batched, size):
    device = batched.device
    h = int(batched.shape[1]) if batched.shape[0] else size
    w = int(batched.shape[2]) if batched.shape[0] else size
    return torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)

def _build_batch_by_indices(batched, indices):
    if len(indices) == 0:
        return batched[0:0]
    return torch.stack([batched[i] for i in indices], dim=0)

def _policy_apply_empty(batched, policy):
    if policy == "return_empty":
        return _empty_like_batch(batched)
    if policy == "black_512":
        return _black_image_batch_like(batched, 512)
    return _pass_through_first(batched)

def _embed_faces(batch_images, app, det_side, resize_mode, info_lines, tag):
    embeddings = []
    faces_per_image = []
    shapes = []
    for idx, img in _iterate_batch(batch_images):
        try:
            bgr, orig_hw = _prepare_for_detection(img, det_side, resize_mode)
            shapes.append((orig_hw[0], orig_hw[1], int(bgr.shape[0]), int(bgr.shape[1])))
            faces = app.get(bgr)
        except Exception as e:
            info_lines.append(f"{tag}[{idx}] preprocess or detect error {repr(e)}")
            faces = []
        faces_per_image.append(len(faces))
        for face in faces:
            vec = face.embedding.astype(np.float32)
            n = np.linalg.norm(vec)
            if n > 0:
                embeddings.append(vec / n)
    if shapes:
        info_lines.append(f"{tag}_shapes={shapes}")
    return embeddings, faces_per_image

class FaceFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_images": ("IMAGE",),
                "candidate_images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "on_empty_matching": (["black_512", "return_empty", "pass_through"],),
                "on_empty_rejected": (["black_512", "return_empty", "pass_through"],),
                "debug": ("BOOLEAN", {"default": False}),
                "model_name": (_PACK_CHOICES,),
                "providers": (["auto(cuda+cpu)", "cpu_only"],),
                "detector_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32}),
                "resize_mode": (["fit_min_side", "fit_longest_side"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("MATCHING", "REJECTED", "DEBUG")
    FUNCTION = "filter"
    CATEGORY = "image/face"
    OUTPUT_NODE = False

    def filter(self, ref_images, candidate_images, threshold, on_empty_matching, on_empty_rejected, debug, model_name, providers, detector_size, resize_mode):
        logs = []
        app = _get_face_app(model_name, providers, detector_size)

        cand_batched = _as_batched_tensor(candidate_images)

        ref_embs, ref_faces = _embed_faces(ref_images, app, detector_size, resize_mode, logs, "refs")
        logs += [
            f"refs_faces_per_image={ref_faces}",
            f"total_ref_embeddings={len(ref_embs)}",
            f"model_name={model_name} providers={providers} det_size={detector_size} resize_mode={resize_mode}",
        ]

        kept_indices = []
        rejected_indices = []
        cand_faces_counts = []
        cand_best_sims = []

        for idx, img in _iterate_batch(cand_batched):
            if not ref_embs:
                cand_faces_counts.append(0)
                cand_best_sims.append(float("-inf"))
                rejected_indices.append(idx)
                continue
            try:
                bgr, _ = _prepare_for_detection(img, detector_size, resize_mode)
                faces = app.get(bgr)
            except Exception:
                cand_faces_counts.append(0)
                cand_best_sims.append(float("-inf"))
                rejected_indices.append(idx)
                continue
            if not faces:
                cand_faces_counts.append(0)
                cand_best_sims.append(float("-inf"))
                rejected_indices.append(idx)
                continue
            cand_faces_counts.append(len(faces))
            best = -1.0
            for face in faces:
                vec = face.embedding.astype(np.float32)
                n = np.linalg.norm(vec)
                if n == 0:
                    continue
                vec /= n
                sims = np.dot(np.stack(ref_embs, axis=0), vec)
                best = max(best, float(np.max(sims)))
            cand_best_sims.append(best)
            if best >= float(threshold):
                kept_indices.append(idx)
            else:
                rejected_indices.append(idx)

        logs.append(f"candidates_faces_per_image={cand_faces_counts}")
        if cand_best_sims:
            logs.append(f"candidates_best_similarity={[round(x,4) if x!=-float('inf') else -1 for x in cand_best_sims]}")
        logs.append(f"threshold={threshold}")

        matching = _build_batch_by_indices(cand_batched, kept_indices)
        rejected = _build_batch_by_indices(cand_batched, rejected_indices)

        if matching.shape[0] == 0:
            matching = _policy_apply_empty(cand_batched, on_empty_matching)
        if rejected.shape[0] == 0:
            rejected = _policy_apply_empty(cand_batched, on_empty_rejected)

        dbg = "\n".join(logs) if debug else ""
        return matching, rejected, dbg

NODE_CLASS_MAPPINGS = {"FaceFilterNode": FaceFilterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceFilterNode": "Face Filter"}
