"""教师课程到Swift数据接口；只转换公开消息和实际图片，不透传私有金标。"""
from pathlib import Path
from ..agent.backend import resize_for_budget
from ..config import Budgets
from ..io import contained_path,file_sha256
from .curriculum_build import load_rgb


def swift_teacher_record(row:dict,root:str|Path) -> dict:
    root=Path(root);budgets=Budgets();messages=[];images=[]
    for message in row['messages']:
        role=message['role']
        if role not in {'system','user','assistant','tool'}:raise ValueError('unsupported_teacher_role')
        if type(message['supervised']) is not bool:raise ValueError('supervision_flag_must_be_bool')
        if role!='assistant' and message['supervised']:raise ValueError('non_assistant_must_not_be_supervised')
        if role=='assistant' and message['images']:raise ValueError('assistant_vision_not_in_this_teacher_protocol')
        text=''
        for ref in message['images']:
            path=contained_path(root,ref['path'])
            if file_sha256(path)!=ref['file_sha256']:raise ValueError('teacher_image_identity_changed')
            image=load_rgb(path)
            cap=budgets.initial_visual_tokens if ref['view_id']=='v0' else budgets.crop_visual_tokens
            images.append(resize_for_budget(image,cap));text+='<image>'
        text+=message['text']
        if role=='tool':
            # Explicitly preserve the model's actual ChatML tool presentation.
            role='user';text='<tool_response>\n'+text+'\n</tool_response>'
        item={'role':role,'content':text}
        if role=='assistant':item['loss']=message['supervised']
        messages.append(item)
    return {'messages':messages,'images':images}
