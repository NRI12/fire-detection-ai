import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

_YOLO_SIZE = 640


class YOLOEigenCAM:
    def __init__(self, yolo_model, device: str = "cpu"):
        self._model = yolo_model.model
        self._device = device
        self._feat: torch.Tensor | None = None

        # CPU has no fp16 support; ensure model is float32 so tensor dtype always matches
        if device == "cpu":
            self._model = self._model.float()

        # SPPF is layer index 9 in YOLOv8 backbone
        target = list(yolo_model.model.model.children())[9]
        target.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out.detach()

    def generate(self, bgr_img: np.ndarray) -> np.ndarray:
        orig_h, orig_w = bgr_img.shape[:2]
        resized = cv2.resize(bgr_img, (_YOLO_SIZE, _YOLO_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb_f32 = rgb.astype(np.float32) / 255.0

        # Match tensor dtype to model weights (float32 on CPU, float16 on CUDA)
        param_dtype = next(self._model.parameters()).dtype
        tensor = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0).to(device=self._device, dtype=param_dtype)
        with torch.no_grad():
            self._model(tensor)

        if self._feat is None:
            return bgr_img.copy()

        # EigenCAM: first principal component of the (H*W, C) activation matrix
        # Cast to float32 for SVD numerical stability and numpy compatibility
        feat = self._feat[0].float()
        C, H_f, W_f = feat.shape
        mat = feat.reshape(C, -1).T

        _, _, Vt = torch.linalg.svd(mat, full_matrices=False)
        cam_flat = mat @ Vt[0]
        cam = cam_flat.cpu().numpy().reshape(H_f, W_f)

        cam = np.maximum(cam, 0)
        cam_range = cam.max() - cam.min()
        cam = (cam - cam.min()) / cam_range if cam_range > 0 else np.zeros_like(cam)

        cam_resized = cv2.resize(cam, (_YOLO_SIZE, _YOLO_SIZE))
        blended = show_cam_on_image(rgb_f32, cam_resized, use_rgb=True)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        if (orig_h, orig_w) != (_YOLO_SIZE, _YOLO_SIZE):
            blended_bgr = cv2.resize(blended_bgr, (orig_w, orig_h))
        return blended_bgr
