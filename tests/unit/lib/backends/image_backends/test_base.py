import asyncio
import base64
import threading
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from lib.backends.artifact_download_guard import (
    IMAGE_ARTIFACT_MAX_BYTES,
    ArtifactDestinationRejectedError,
    ArtifactTooLargeError,
)
from lib.backends.image_backends.base import (
    ImageCapability,
    ImageGenerationRequest,
    ImageGenerationResult,
    ReferenceImage,
    download_image_to_path,
    save_image_from_response_item,
)
from tests.http_capture import capture_http


def test_image_capability_is_str_enum():
    assert ImageCapability.TEXT_TO_IMAGE == "text_to_image"
    assert ImageCapability.IMAGE_TO_IMAGE == "image_to_image"


def test_reference_image_carries_only_the_path():
    ref = ReferenceImage(path="/tmp/test.png")
    assert ref.path == "/tmp/test.png"
    assert [f.name for f in fields(ref)] == ["path"]


def test_image_generation_request_defaults():
    req = ImageGenerationRequest(prompt="hello", output_path=Path("/tmp/out.png"))
    assert req.aspect_ratio == "9:16"
    assert req.image_size is None
    assert req.reference_images == []
    assert req.project_name is None
    assert req.seed is None


def test_image_generation_result():
    result = ImageGenerationResult(
        image_path=Path("/tmp/out.png"),
        provider="grok",
        model="grok-imagine-image",
    )
    assert result.image_uri is None
    assert result.seed is None
    assert result.usage_tokens is None


async def test_download_image_writes_the_response_body(tmp_path: Path):
    output = tmp_path / "out.png"
    with capture_http() as router:
        router.get("http://203.0.113.7/a.png").mock(return_value=httpx.Response(200, content=b"\x89PNG"))
        await download_image_to_path("http://203.0.113.7/a.png", output)
    assert output.read_bytes() == b"\x89PNG"


async def test_download_image_rejects_link_local_destination(tmp_path: Path):
    output = tmp_path / "out.png"
    with capture_http() as router:
        route = router.get("http://169.254.169.254/a.png").mock(return_value=httpx.Response(200))
        with pytest.raises(ArtifactDestinationRejectedError):
            await download_image_to_path("http://169.254.169.254/a.png", output)
    assert route.call_count == 0
    assert not output.exists()


async def test_download_image_aborts_when_declared_size_exceeds_the_image_limit(tmp_path: Path):
    output = tmp_path / "out.png"
    oversized = {"Content-Length": str(IMAGE_ARTIFACT_MAX_BYTES + 1)}
    with capture_http() as router:
        router.get("http://203.0.113.7/a.png").mock(return_value=httpx.Response(200, headers=oversized, content=b""))
        with pytest.raises(ArtifactTooLargeError):
            await download_image_to_path("http://203.0.113.7/a.png", output)
    assert not output.exists()
    assert not (tmp_path / f"{output.name}.part").exists()


class _GatedWritePath(Path):
    """写入被闸门挡住的 ``output_path``：观测取消与线程写入的交错，不 patch 生产符号。"""

    def attach(self, started: threading.Event, release: threading.Event) -> "_GatedWritePath":
        self._started = started
        self._release = release
        return self

    def write_bytes(self, data) -> int:
        self._started.set()
        self._release.wait(5)
        return super().write_bytes(data)


async def test_save_b64_item_writes_decoded_bytes(tmp_path: Path):
    raw = b"\x89PNG\r\n\x1a\nfake-image"
    item = SimpleNamespace(b64_json=base64.b64encode(raw).decode("ascii"), url=None)
    output = tmp_path / "out.png"
    await save_image_from_response_item(item, output)
    assert output.read_bytes() == raw


async def test_save_b64_item_requires_b64_or_url(tmp_path: Path):
    with pytest.raises(ValueError, match="既无 b64_json 也无 url"):
        await save_image_from_response_item(SimpleNamespace(b64_json=None, url=None), tmp_path / "out.png")


async def test_cancel_during_b64_write_still_completes(tmp_path: Path):
    """落账段不可中断：写盘线程被取消挡住时，任务不提前结束；闸门放开后照常完成，不抛 CancelledError。

    旧实现（``run_sync_transaction``）写盘本身会跑完，但随后把取消重抛给调用方，
    其后的转码与产物提交被跳过——该用例在旧实现上以 CancelledError 失败。
    """
    raw = b"\x89PNG\r\n\x1a\nfake-image"
    item = SimpleNamespace(b64_json=base64.b64encode(raw).decode("ascii"), url=None)
    started = threading.Event()
    release = threading.Event()
    output = _GatedWritePath(tmp_path / "out.png").attach(started, release)

    task = asyncio.create_task(save_image_from_response_item(item, output))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    # 线程往返把控制权交还事件循环：取消已送达，但写盘未结算前任务不得结束
    await asyncio.to_thread(lambda: None)
    assert not task.done()
    release.set()
    await task
    assert output.read_bytes() == raw
