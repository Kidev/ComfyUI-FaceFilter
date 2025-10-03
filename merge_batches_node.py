import torch
import numpy as np

def _as_batched_tensor(imgs):
    if isinstance(imgs, torch.Tensor):
        return imgs if imgs.ndim == 4 else imgs.unsqueeze(0)
    if isinstance(imgs, (list, tuple)):
        ts = []
        for im in imgs:
            if not isinstance(im, torch.Tensor):
                # convert numpy HWC uint8 or float to torch float32 0..1
                if isinstance(im, np.ndarray):
                    arr = im.astype(np.float32)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim != 3 or arr.shape[-1] < 3:
                        raise ValueError("Unsupported numpy image shape")
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.max() > 1.0: arr = arr / 255.0
                    im = torch.from_numpy(arr)
                else:
                    raise ValueError("Unsupported IMAGE element type")
            ts.append(im if im.ndim == 3 else im.squeeze(0))
        return torch.stack(ts, dim=0) if ts else torch.zeros((0, 64, 64, 3), dtype=torch.float32)
    raise ValueError("Unsupported IMAGE container")

def _is_black_or_empty(img_tensor):
    if not isinstance(img_tensor, torch.Tensor): return True
    if img_tensor.numel() == 0: return True
    return bool(torch.all(img_tensor == 0))

class MergeImageBatches:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch1": ("IMAGE",),
                "batch2": ("IMAGE",),
                "prefer": (["batch1", "batch2"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("MERGED",)
    FUNCTION = "merge"
    CATEGORY = "image/util"

    def merge(self, batch1, batch2, prefer):
        b1 = _as_batched_tensor(batch1)
        b2 = _as_batched_tensor(batch2)
        n = min(int(b1.shape[0]), int(b2.shape[0]))
        items = []
        for i in range(n):
            a = b1[i]; b = b2[i]
            a_bad = _is_black_or_empty(a)
            b_bad = _is_black_or_empty(b)
            if a_bad and not b_bad:
                items.append(b)
            elif b_bad and not a_bad:
                items.append(a)
            else:
                items.append(a if prefer == "batch1" else b)
        return (torch.stack(items, dim=0) if items else b1[0:0],)

NODE_CLASS_MAPPINGS = {"MergeImageBatches": MergeImageBatches}
NODE_DISPLAY_NAME_MAPPINGS = {"MergeImageBatches": "Merge image batches"}
