"""Official RIFE IFNet inference model with CPU-safe warp and weight resolver."""

from __future__ import annotations

import hashlib
import html
import http.cookiejar
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchRIFEModelSpec:
    key: str
    file_id: str
    archive_name: str
    checkpoint_relpath: str
    checkpoint_sha256: str
    description: str


KNOWN_TORCH_RIFE_MODELS: dict[str, TorchRIFEModelSpec] = {
    "paper_v6": TorchRIFEModelSpec(
        key="paper_v6",
        file_id="1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX",
        archive_name="RIFE_trained_v6.zip",
        checkpoint_relpath="train_log/flownet.pkl",
        checkpoint_sha256="2b48d6f7c7c09c109ad50372e9d9dc722a75859dc20e8832a62826ecede35ad3",
        description="Official ECCV2022-RIFE paper checkpoint (v6 package, flownet.pkl)",
    ),
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "enhanced" / "models" / "rife_torch"
_GRID_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_google_drive_file(file_id: str, dest: Path) -> None:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    request_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    with opener.open(request_url, timeout=60) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()

    if "text/html" not in content_type:
        dest.write_bytes(body)
        return

    text = body.decode("utf-8", errors="replace")
    form_action = re.search(r'action="([^"]+)"', text)
    hidden_inputs = re.findall(r'name="([^"]+)" value="([^"]+)"', text)
    if not form_action or not hidden_inputs:
        raise RuntimeError("Google Drive response did not contain a download form")

    action_url = html.unescape(form_action.group(1))
    params = {key: html.unescape(value) for key, value in hidden_inputs}
    download_url = f"{action_url}?{urllib.parse.urlencode(params)}"

    with opener.open(download_url, timeout=300) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def ensure_torch_rife_checkpoint(
    model_name: str = "paper_v6",
    model_file: str | None = None,
    model_dir: str | None = None,
) -> Path:
    """Return a local `flownet.pkl` path for the requested torch RIFE model."""
    if model_file:
        path = Path(model_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RIFE torch model file not found: {path}")
        return path

    if model_dir:
        path = Path(model_dir).expanduser().resolve() / "flownet.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"RIFE torch model dir missing flownet.pkl: {path.parent}")
        return path

    try:
        spec = KNOWN_TORCH_RIFE_MODELS[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown torch RIFE model: {model_name!r}") from exc

    target_dir = _DEFAULT_MODELS_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = target_dir / "flownet.pkl"
    if checkpoint.is_file():
        actual = _sha256_file(checkpoint)
        if actual == spec.checkpoint_sha256:
            return checkpoint
        checkpoint.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="rife_torch_") as td:
        tmp_dir = Path(td)
        archive = tmp_dir / spec.archive_name
        _download_google_drive_file(spec.file_id, archive)

        with zipfile.ZipFile(archive) as zf:
            try:
                member = next(
                    name for name in zf.namelist()
                    if name.rstrip("/") == spec.checkpoint_relpath
                )
            except StopIteration as exc:
                raise RuntimeError(
                    f"Official archive for {model_name!r} did not contain {spec.checkpoint_relpath}"
                ) from exc

            extracted = tmp_dir / "flownet.pkl"
            with zf.open(member) as src, extracted.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)

        actual = _sha256_file(extracted)
        if actual != spec.checkpoint_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for torch RIFE model {model_name!r}: "
                f"expected {spec.checkpoint_sha256}, got {actual}"
            )

        shutil.move(str(extracted), checkpoint)

    return checkpoint


def warp(ten_input: torch.Tensor, ten_flow: torch.Tensor) -> torch.Tensor:
    device = ten_flow.device
    key = (str(device), tuple(ten_flow.shape))
    if key not in _GRID_CACHE:
        horizontal = torch.linspace(-1.0, 1.0, ten_flow.shape[3], device=device).view(
            1, 1, 1, ten_flow.shape[3]
        ).expand(ten_flow.shape[0], -1, ten_flow.shape[2], -1)
        vertical = torch.linspace(-1.0, 1.0, ten_flow.shape[2], device=device).view(
            1, 1, ten_flow.shape[2], 1
        ).expand(ten_flow.shape[0], -1, -1, ten_flow.shape[3])
        _GRID_CACHE[key] = torch.cat([horizontal, vertical], 1)

    flow = torch.cat(
        [
            ten_flow[:, 0:1] / ((ten_input.shape[3] - 1.0) / 2.0),
            ten_flow[:, 1:2] / ((ten_input.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    grid = (_GRID_CACHE[key] + flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=ten_input,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1,
         padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


def deconv(in_planes: int, out_planes: int, kernel_size: int = 4,
           stride: int = 2, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


_C = 16


class Contextnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2(3, _C)
        self.conv2 = Conv2(_C, 2 * _C)
        self.conv3 = Conv2(2 * _C, 4 * _C)
        self.conv4 = Conv2(4 * _C, 8 * _C)

    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = Conv2(17, 2 * _C)
        self.down1 = Conv2(4 * _C, 4 * _C)
        self.down2 = Conv2(8 * _C, 8 * _C)
        self.down3 = Conv2(16 * _C, 16 * _C)
        self.up0 = deconv(32 * _C, 8 * _C)
        self.up1 = deconv(16 * _C, 4 * _C)
        self.up2 = deconv(8 * _C, 2 * _C)
        self.up3 = deconv(4 * _C, _C)
        self.conv = nn.Conv2d(_C, 3, 3, 1, 1)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        warped_img0: torch.Tensor,
        warped_img1: torch.Tensor,
        mask: torch.Tensor,
        flow: torch.Tensor,
        c0: list[torch.Tensor],
        c1: list[torch.Tensor],
    ) -> torch.Tensor:
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        return torch.sigmoid(self.conv(x))


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow_out, mask


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(17, c=150)
        self.block2 = IFBlock(17, c=90)
        self.block_tea = IFBlock(20, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(
        self,
        x: torch.Tensor,
        scale: list[float] | tuple[float, float, float] = (4, 2, 1),
        timestep: float = 0.5,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        del timestep
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow_list: list[torch.Tensor] = []
        merged: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        for idx, block in enumerate((self.block0, self.block1, self.block2)):
            if flow is not None and mask is not None:
                flow_delta, mask_delta = block(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                    scale=scale[idx],
                )
                flow = flow + flow_delta
                mask = mask + mask_delta
            else:
                flow, mask = block(torch.cat((img0, img1), 1), None, scale=scale[idx])
            mask_sigmoid = torch.sigmoid(mask)
            mask_list.append(mask_sigmoid)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_sigmoid + warped_img1 * (1 - mask_sigmoid))

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        residual = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        merged[2] = torch.clamp(merged[2] + (residual[:, :3] * 2 - 1), 0, 1)
        return flow_list, mask_list[2], merged


class OfficialRIFEInterpolator:
    """Official IFNet inference wrapper for numpy RGB frames."""

    def __init__(self, checkpoint: Path, device: str = "cpu", cpu_threads: int = 0):
        self.device = torch.device(device)
        if self.device.type == "cpu" and cpu_threads > 0:
            torch.set_num_threads(cpu_threads)

        try:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=self.device)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported RIFE checkpoint payload in {checkpoint}")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        normalized = {
            key.replace("module.", ""): value
            for key, value in state.items()
            if isinstance(key, str)
        }

        model = IFNet()
        model.load_state_dict(normalized, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        h, w = frame0.shape[:2]
        t0 = self._frame_to_tensor(frame0)
        t1 = self._frame_to_tensor(frame1)

        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        if ph != h or pw != w:
            padding = (0, pw - w, 0, ph - h)
            t0 = F.pad(t0, padding)
            t1 = F.pad(t1, padding)

        with torch.inference_mode():
            merged = self.model(torch.cat((t0, t1), 1), [4, 2, 1], timestep=0.5)[2][2]

        merged = merged[:, :, :h, :w]
        out = (
            merged[0]
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return out

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(frame)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")
