from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from ..io import finite, identity

_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class QARecord:
    sample_id: str
    source: str
    source_revision: str
    native_qa_id: str
    source_split: str
    image_id: str
    image_path: str
    image_sha256: str
    image_width: int
    image_height: int
    question: str
    answers: list[str]
    image_group_id: str
    project_split: str
    task_type: str
    product_family_id: str | None = None
    evidence_box_xyxy: list[float] | None = None
    annotation_version: str = "unreviewed"
    evidence_source: str = "none"
    quality_flags: list[str] = field(default_factory=list)
    provenance_kind: str = "observed"
    native_gold_match: str = "unverified"

    def validate(self, require_evidence: bool = False) -> "QARecord":
        for key in ("sample_id", "source", "source_revision", "native_qa_id", "image_id", "question", "image_group_id", "task_type",'source_split','image_path','annotation_version','evidence_source'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"缺少 {key}")
        if not isinstance(self.image_sha256,str) or not _SHA.fullmatch(self.image_sha256):
            raise ValueError("原图 SHA256 格式错误")
        for value in (self.image_width, self.image_height):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("自然图像尺寸必须为正整数")
        if self.project_split not in {"train", "dev", "test", "diagnostic", "quarantine"}:
            raise ValueError("非法 project_split")
        if self.provenance_kind not in {"observed", "synthetic_fixture"}:
            raise ValueError("目标设定不能作为数据记录")
        if not isinstance(self.answers,list) or not self.answers or not all(isinstance(x, str) and x.strip() for x in self.answers):
            raise ValueError("至少提供一个可追溯答案")
        if not isinstance(self.quality_flags,list) or not all(isinstance(x,str) for x in self.quality_flags):
            raise ValueError('quality_flags必须为字符串列表')
        if self.project_split == "train" and self.source_split.lower() in {"test", "validation", "val", "testmini"}:
            raise ValueError("上游评测样本不得转为训练")
        if require_evidence and self.evidence_box_xyxy is None:
            raise ValueError("任务要求独立紧框标注；整商品框不能替代文字证据")
        if self.evidence_box_xyxy is not None:
            if len(self.evidence_box_xyxy) != 4:
                raise ValueError("紧框必须为 xyxy")
            x0, y0, x1, y1 = [finite(v, "evidence_box") for v in self.evidence_box_xyxy]
            if not (0 <= x0 < x1 <= self.image_width and 0 <= y0 < y1 <= self.image_height):
                raise ValueError("紧框超出自然原图")
        return self

    def actor_input(self) -> dict:
        """白名单构造；不透传答案、金框、家族、质量标记等私有字段。"""
        return {"task_id": self.sample_id, "question": self.question,
                "image_id": self.image_id, "image_path": self.image_path,
                "image_sha256": self.image_sha256}

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_native_qa(row: dict, source: str, revision: str) -> dict:
    """支持显式 QA 或常见 LLaVA 单轮字段；不从合成推理尾部猜原生金标。"""
    question = row.get("question")
    target = row.get("target_response")
    if question is None:
        conv = row.get("conversations", row.get("messages", []))
        users = [m.get("value", m.get("content")) for m in conv if m.get("from", m.get("role")) in {"human", "user"}]
        answers = [m.get("value", m.get("content")) for m in conv if m.get("from", m.get("role")) in {"gpt", "assistant"}]
        if len(users) != 1 or len(answers) != 1 or not isinstance(users[0], str):
            raise ValueError("当前适配器只归一化单轮图文问答")
        question, target = users[0].replace("<image>", "").strip(), answers[0]
    images = row.get("images", row.get("image"))
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("仅支持单图，images 字符串需显式转列表")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("空题干")
    choices = row.get("choices", [])
    gold = row.get("native_answer")
    return {"sample_id": identity([source, revision, str(row.get("id", "")), question, choices]),
            "source": source, "source_revision": revision,
            "native_qa_id": str(row.get("id", "")), "question": question,
            "choices": choices, "image_path": images[0],
            "native_answer": gold, "target_response": target,
            "native_gold_match": "provided_not_verified" if gold is not None else "unverified"}


def match_native_gold(candidate: dict, originals: list[dict]) -> dict:
    """原始键仍须核对题干/图身份；选项重排按语义映射。"""
    def norm(text):
        return " ".join(str(text).split())
    matches = [r for r in originals if r["source"] == candidate["source"]
               and r["image_id"] == candidate["image_id"]
               and norm(r["question"]) == norm(candidate["question"])
               and sorted(r.get("choices", [])) == sorted(candidate.get("choices", []))]
    if len(matches) != 1:
        raise ValueError("原生 QA 无唯一匹配；禁止只按图片套答案")
    r = matches[0]
    answer = r["answer"]
    choices = r.get("choices", [])
    if choices and isinstance(answer, str) and len(answer) == 1 and "A" <= answer <= "Z":
        index = ord(answer) - ord("A")
        if index >= len(choices):
            raise ValueError("原选项字母越界")
        answer = choices[index]
    mapped = candidate.get("choices", [])
    if mapped:
        if mapped.count(answer) != 1:
            raise ValueError("重排选项无唯一语义答案")
        answer = chr(ord("A") + mapped.index(answer))
    return {"answer": answer, "native_qa_id": r["native_qa_id"], "native_gold_match": "question_choices_image_checked"}
