from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from ..io import identity


def image_fingerprints(path: str) -> dict:
    """pHash 仅作候选；EXIF→RGB(白底) 的规范像素另行记录。"""
    from PIL import Image, ImageOps
    import numpy as np
    from ..io import file_sha256
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        rgb = background.convert("RGB")
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + b"|RGB|" + rgb.tobytes()).hexdigest()
        values = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float64)
    n = np.arange(32)
    matrix = np.cos(np.pi / 32 * (n[None, :] + 0.5) * n[:, None])
    dct = matrix @ values @ matrix.T
    low = dct[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {"file_sha256": file_sha256(path), "pixel_sha256": pixel_hash,'width':rgb.width,'height':rgb.height,
            "phash64": f"{phash:016x}", "version": "exif_rgb_white_dct32_v1"}


class CandidateBudgetExceeded(ValueError):
    def __init__(self,pairs,limit):
        super().__init__('phash_candidate_pair_budget_exceeded')
        self.pairs=pairs[:limit]
        self.lower_bound=len(pairs)


def phash_candidates(rows: list[dict], distance: int = 8,max_pairs:int|None=None) -> list[dict]:
    """CPU BK-tree 候选检索；不把相似包装自动认定为同源。"""
    if isinstance(distance,bool) or not isinstance(distance,int) or not 0<=distance<=64:raise ValueError('pHash距离必须为0到64')
    if max_pairs is not None and (isinstance(max_pairs,bool) or not isinstance(max_pairs,int) or max_pairs<1):raise ValueError('pHash候选预算必须为正整数')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('pHash图片身份重复')
    if any(not isinstance(r['phash64'],str) or not re.fullmatch('[0-9a-fA-F]{16}',r['phash64']) for r in rows):raise ValueError('pHash必须是64位指纹')
    tree = None
    result = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        value = int(row["phash64"], 16)
        if tree is None:
            tree = [value, [row["image_id"]], {}]
            continue
        stack = [tree]
        while stack:
            node = stack.pop()
            d = (value ^ node[0]).bit_count()
            if d <= distance:
                for uid in node[1]:
                    result.append({"left":uid,"right":row['image_id'],'hamming':d,'confirmed':False})
                    if max_pairs is not None and len(result)>max_pairs:
                        raise CandidateBudgetExceeded(result,max_pairs)
            stack.extend(child for edge, child in node[2].items() if d - distance <= edge <= d + distance)
        node = tree
        while True:
            d = (value ^ node[0]).bit_count()
            if d == 0:
                node[1].append(row["image_id"])
                break
            if d not in node[2]:
                node[2][d] = [value, [row["image_id"]], {}]
                break
            node = node[2][d]
    return result


def build_groups(images: list[dict], edges: list[dict], protected: set[str] | None = None) -> dict:
    uids = [r["image_id"] for r in images]
    if len(set(uids)) != len(uids):
        raise ValueError("图像身份重复")
    parent = {uid: uid for uid in uids}
    def root(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid
    def union(a, b):
        if a not in parent or b not in parent:
            raise ValueError("关系引用未知图像")
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    buckets = {}
    for row in sorted(images, key=lambda r: r["image_id"]):
        for key in ("file_sha256", "pixel_sha256", "product_family_id"):
            value = row.get(key)
            if value:
                token = (key, value)
                if token in buckets:
                    union(row["image_id"], buckets[token])
                else:
                    buckets[token] = row["image_id"]
    for edge in edges:
        if edge.get("confirmed") is True and edge.get("basis") in {"known_crop", "geometry_verified", "human_reviewed"}:
            union(edge["left"], edge["right"])
    groups = defaultdict(list)
    for uid in sorted(parent):
        groups[root(uid)].append(uid)
    protected = protected or set()
    if not protected <= set(parent):
        raise ValueError("保护图像必须参与分组")
    mapping, quarantined = {}, []
    for representative, members in sorted(groups.items()):
        group = identity(["image-family-group-v1", representative])
        mapping.update({uid: group for uid in members})
        if protected.intersection(members):
            quarantined.append(group)
    return {"mapping": mapping, "quarantined_groups": quarantined,
            "components": list(groups.values()), "version": "image-family-group-v1"}


def split_and_sample(rows: list[dict], counts: dict[str, int], seed: int = 42) -> dict:
    """整组只分配一次；同组多余 QA 可不选，不能跨 split 补齐。"""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("protected") or row.get("quality_flags") or str(row.get("source_split", "train")).lower() in {"test", "val", "validation", "testmini"}:
            raise ValueError("上游评测、保护或未通过质量检查的记录不得进入新划分候选")
        if row["sample_id"] in seen:
            raise ValueError("QA 身份重复")
        seen.add(row["sample_id"])
        groups[row["image_group_id"]].append(row)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("划分配额必须为非负整数")
    if not counts or set(counts)-{'train','dev','test','diagnostic'}:
        raise ValueError('只支持明确的train/dev/test/diagnostic划分名')
    if len(rows)<sum(counts.values()):
        raise ValueError('insufficient_rows_for_requested_counts')
    # 先为较小的dev/test预留整组容量，避免先填train浪费最后一组后误报不足。
    def reserve(available,needed,leave_min):
        if not needed:return []
        capacity=sum(len(groups[g]) for g in available)-leave_min
        if capacity<needed:return None
        horizon=min(capacity,needed+max((len(groups[g]) for g in available),default=0)-1)
        states={0:None}
        for group in available:
            size=len(groups[group])
            for total,parent in list(states.items()):
                new=total+size
                if new<=horizon and new not in states:
                    states[new]=(group,parent)
            if needed in states:break
        choices=[n for n in states if needed<=n<=capacity]
        if not choices:return None
        node=states[min(choices)];selected=[]
        while node is not None:
            group,node=node;selected.append(group)
        return selected
    primary='train' if 'train' in counts else max(counts,key=lambda s:(counts[s],s))
    for attempt in range(6):
        available=sorted(groups,key=lambda g:identity([seed,attempt,g]));assigned={};output={key:[] for key in counts}
        other=sorted((s for s in counts if s!=primary),key=lambda s:(counts[s],s))
        if attempt%2:other.reverse()
        feasible=True
        for i,split in enumerate(other):
            leave=counts[primary]+sum(counts[s] for s in other[i+1:])
            chosen=reserve(available,counts[split],leave)
            if chosen is None:
                feasible=False;break
            need=counts[split]
            for group in chosen:
                assigned[group]=split
                take=sorted(groups[group],key=lambda r:r['sample_id'])[:need]
                output[split].extend({**r,'project_split':split} for r in take);need-=len(take)
            used=set(chosen);available=[g for g in available if g not in used]
        if not feasible:continue
        need=counts[primary]
        for group in available:
            if need<=0:break
            assigned[group]=primary
            take=sorted(groups[group],key=lambda r:r['sample_id'])[:need]
            output[primary].extend({**r,'project_split':primary} for r in take);need-=len(take)
        if need:continue
        return {'splits':output,'group_assignment':assigned,'seed':seed,'assignment_attempts':attempt+1,
                'algorithm':'bounded_group_capacity_reservation_v2','labels_or_model_scores_used_for_assignment':False}
    raise ValueError('group_assignment_not_found_within_bounded_search; 不拆家族、不复制，需检查分组或扩大候选')
