"""现有本地 Qwen3-VL/LoRA 的可选只推理后端。导入此模块不加载 Torch。"""
from __future__ import annotations

import math
import time
from pathlib import Path
from ..config import Budgets
from ..io import file_sha256, identity


def resize_for_budget(image, cap: int, factor: int = 32):
    from PIL import Image
    width, height = image.size
    scale = min(1., math.sqrt(cap * factor * factor / (width * height)))
    w = max(factor, int(width * scale) // factor * factor)
    h = max(factor, int(height * scale) // factor * factor)
    while w // factor * (h // factor) > cap:
        if w >= h and w > factor:
            w -= factor
        elif h > factor:
            h -= factor
        else:
            raise ValueError("视觉预算太小")
    return image.resize((w, h), Image.Resampling.LANCZOS)


class LocalQwenBackend:
    def __init__(self, model_path: str, adapter_path: str | None = None, device: str = "cpu", seed: int = 42,
                 budgets: Budgets | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.seed, self.budgets = torch, seed, budgets or Budgets()
        if not Path(model_path).is_dir():
            raise ValueError("只接受已存在的本地基座目录；不自动下载模型")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if device == "cpu":
            dtype, device_map = torch.float32, "cpu"
        elif device.startswith("cuda") and torch.cuda.is_available():
            dtype, device_map = torch.bfloat16, device
        else:
            raise ValueError("指定设备不可用")
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
                          trust_remote_code=False, torch_dtype=dtype, device_map=device_map).eval()
        if adapter_path:
            from peft import PeftModel
            weights = Path(adapter_path) / "adapter_model.safetensors"
            if not weights.is_file():
                raise ValueError("现有 LoRA 缺少 adapter_model.safetensors")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False, local_files_only=True,
                                                  autocast_adapter_dtype=False).eval()
            adapter_sha = file_sha256(weights)
        else:
            adapter_sha = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.identity = {"backend": "local_transformers_qwen3vl", "adapter_sha256": adapter_sha,
                         "config_sha256": file_sha256(Path(model_path) / "config.json"),
                         "seed": seed, "device": device, "training": False,
                         "full_base_weights_sha256": "not_recomputed_by_this_backend"}
        self.inputs = None

    def prepare(self, messages: list[dict]) -> dict:
        from PIL import Image
        images, clean = [], []
        for message in messages:
            content = []
            for item in message["content"]:
                if item["type"] == "image":
                    cap = self.budgets.initial_visual_tokens if item["view_id"] == "v0" else self.budgets.crop_visual_tokens
                    image = resize_for_budget(item["image"], cap)
                    images.append(image)
                    content.append({"type": "image", "image": image})
                else:
                    content.append({"type": "text", "text": item["text"]})
            clean.append({"role": message["role"], "content": content})
        text = self.processor.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
        self.inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")
        grid = self.inputs.get("image_grid_thw")
        if grid is None or len(grid) != len(images):
            raise ValueError("processor 图像数不一致；禁止仅以文本路径当观察")
        merge = getattr(self.processor.image_processor, "merge_size", None)
        if not isinstance(merge, int) or merge <= 0:
            raise ValueError("无法确认实际视觉 merge_size")
        counts = [int(row.prod().item()) // (merge * merge) for row in grid]
        for count, message_item in zip(counts, [i for m in messages for i in m["content"] if i["type"] == "image"]):
            cap = self.budgets.initial_visual_tokens if message_item["view_id"] == "v0" else self.budgets.crop_visual_tokens
            if count > cap:
                raise ValueError("真实 processor 超出每图上限，需修订分辨率预处理")
        return {"context_tokens": int(self.inputs["input_ids"].shape[-1]), "visual_counts": counts,
                "input_ids_sha256": identity(self.inputs["input_ids"].tolist()),
                "image_grid_thw": grid.tolist(), "processor_verified": True}

    def generate(self, max_new_tokens: int, do_sample: bool = False) -> dict:
        if self.inputs is None or max_new_tokens <= 0:
            raise ValueError("先调用 prepare 并完成环境准入")
        torch = self.torch
        inputs = self.inputs.to(self.model.device)
        start = time.perf_counter()
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kwargs.update(temperature=1., top_p=1., top_k=0, min_p=0., repetition_penalty=1.)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        tokens = output[0, inputs["input_ids"].shape[-1]:].tolist()
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        ended = bool(tokens and tokens[-1] in eos)
        return {"text": text, "token_ids": tokens, "finish_reason": "stop" if ended else "length",
                "latency_seconds": time.perf_counter() - start,
                "cpu_rng_sha256": identity(torch.get_rng_state().tolist()),
                "cuda_rng_sha256": [identity(x.tolist()) for x in torch.cuda.get_rng_state_all()] if self.model.device.type == "cuda" else []}


def cpu_processor_preflight(model_path: str, output: str | Path) -> dict:
    """实际processor+真实fixture crop；明确不加载模型/不生成token。"""
    import importlib.metadata
    from PIL import Image
    from transformers import AutoProcessor
    from .environment import CropEnvironment
    from ..io import canonical
    start=time.perf_counter()
    dest=Path(output)
    dest.mkdir(parents=True,exist_ok=True)
    image=dest/"synthetic_processor_fixture.png"
    Image.new("RGB",(1200,1600),"white").save(image)
    environment=CropEnvironment({"task_id":"synthetic-processor-fixture","question":"Read the fixture image.",
                                 "image_id":"synthetic-fixture","image_path":image.name,"image_sha256":file_sha256(image)},
                                dest,dest/"views")
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    backend=object.__new__(LocalQwenBackend)
    backend.processor=processor
    backend.budgets=Budgets()
    backend.inputs=None
    image_token=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    cases=[]
    for count in range(4):
        if count:
            environment.step(canonical({"tool":"crop","view_id":"v0","box":[500,600,950,850]}),
                             [900000+count],"synthetic_fixture_end")
        prepared=backend.prepare(environment.messages)
        placeholders=int((backend.inputs["input_ids"]==image_token).sum().item())
        if placeholders!=sum(prepared["visual_counts"]):
            raise ValueError("真实image占位token与grid计数不匹配")
        if len(prepared["visual_counts"])!=count+1:
            raise ValueError("成功crop没有真实增加图片")
        room=environment.admit_processor(prepared["context_tokens"],prepared["visual_counts"])
        if room<=0:
            raise ValueError("正常短fixture被预算拒绝")
        cases.append({"views":count+1,**prepared,"image_placeholder_count":placeholders,"admitted_output_tokens":room})
    environment.messages.append({"role":"user","content":[{"type":"text","text":" x"*7000}]})
    overflow=backend.prepare(environment.messages)
    if overflow["context_tokens"]<=environment.budgets.context_tokens:
        raise ValueError("溢出fixture没有真实越过context限制")
    if environment.admit_processor(overflow["context_tokens"],overflow["visual_counts"])!=0 or not environment.done:
        raise ValueError("实际长前缀未触发停止")
    versions={}
    for module in ("torch","transformers","Pillow","peft"):
        try:versions[module]=importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:versions[module]=None
    return {"status":"passed","scope":"real_CPU_processor_and_true_pixel_crop_on_isolated_synthetic_fixture",
            "data_kind":"synthetic_fixture_not_model_evaluation","fixture_action_token_ids":True,
            "cases":cases,"cases_checked":len(cases),"overflow_context_tokens":overflow["context_tokens"],
            "context_overflow_stopped":True,"processor_class":type(processor).__name__,"versions":versions,
            "eos_token_id":processor.tokenizer.eos_token_id,"cpu_wall_seconds":time.perf_counter()-start,
            "model_weights_loaded":False,"model_generate_calls":0,"optimizer_steps":0,
            "native_training_masks_verified":False,"full_8b_generation_verified":False,"gpu_model_seconds":0}
