"""本地来源适配：RPC 整商品标注只产出候选，不自动升级成 tight-box QA。"""
from __future__ import annotations

from ..io import identity
from .geometry_bridge import checked_text_region


def rpc_exemplar_candidates(coco: dict, revision: str) -> list[dict]:
    if not revision:
        raise ValueError("来源版本必需")
    image_ids = [image["id"] for image in coco["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("RPC 图像身份重复")
    images = {image["id"]: image for image in coco["images"]}
    categories = {category["id"]: category for category in coco.get("categories", [])}
    output = []
    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        from ..agent.geometry import box
        box([x, y, x + w, y + h], image["width"], image["height"])
        output.append({"candidate_id": identity(["RPC", revision, annotation["id"]]), "source": "RPC",
                       "source_revision": revision, "source_split": "exemplar_train",
                       "image_id": str(image["id"]), "image_path": image["file_name"],
                       "width": image["width"], "height": image["height"],
                       "whole_product_box_xyxy": [x, y, x + w, y + h],
                       "category_id": annotation["category_id"],
                       "category": categories.get(annotation["category_id"], {}).get("name"),
                       "qa_status": "requires_field_transcription_tight_box_and_family_annotation"})
    return output


def unitail_reviewed_region(export: dict, revision: str) -> dict:
    """读取人工审阅的区域导出，不猜测官方压缩包中不同组件的异构布局。"""
    required = {"image_id", "image_path", "width", "height", "region_id", "polygon", "transcription", "component", "review_receipt"}
    if not required <= set(export) or not export["review_receipt"]:
        raise ValueError("缺少 Unitail 区域导出字段或审阅收据")
    if export["component"] != "gallery_train":
        raise ValueError("官方测试组件只能作保护/评测，不进训练候选")
    rectangle = checked_text_region(export["polygon"], export["width"], export["height"])
    if not export["transcription"].strip() or export["transcription"].strip() in {"*", "###", "???"}:
        raise ValueError("不可读文字不生成金答案")
    return {"source": "Unitail-OCR", "source_revision": revision,
            "candidate_id": identity(["Unitail-OCR", revision, export["image_id"], export["region_id"]]),
            "image_id": export["image_id"], "image_path": export["image_path"],
            "transcription": export["transcription"], "region_box_xyxy": rectangle,
            "qa_status": "requires_unique_question_semantics_and_product_family_review",
            "review_receipt": export["review_receipt"]}
