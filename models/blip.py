import warnings

warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        """
        初始化 BLIP 编码-解码器模型
        """
        super().__init__()

        # 1. 视觉编码器 (ViT)
        self.visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=768, depth=12,
                                                num_heads=12,
                                                qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        # 2. 文本分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # 3. 文本解码器 (X-BERT)
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = 768
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        # =================================================================
        # [论文魔改位置 1]：在这里初始化您的三个创新模块
        # 例如：
        # self.module_A = NovelModuleA(...) # 增强局部特征
        # self.module_B = NovelModuleB(...) # 跨模态融合优化
        # self.module_C = NovelModuleC(...) # 解码器引导
        # =================================================================

    def forward(self, image, caption):
        """
        前向传播函数 (微调训练时调用)
        """
        # 视觉特征提取
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # =================================================================
        # [论文魔改位置 2]：在这里应用您的 Module A 和 Module B
        # 例如：
        # image_embeds = self.module_A(image_embeds)
        # image_embeds, _ = self.module_B(image_embeds, text_embeds)
        # =================================================================

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        # =================================================================
        # [论文魔改位置 3]：在这里应用您的 Module C 引导文本生成
        # =================================================================

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss = decoder_output.loss

        return loss

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        """
        生成描述函数 (测试评估时调用)
        """
        image_embeds = self.visual_encoder(image)

        # [魔改提示] 如果前向传播中改了特征，这里生成时也要做对应修改！

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids, max_length=max_length, min_length=min_length,
                                                 do_sample=True, top_p=top_p, num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids, max_length=max_length, min_length=min_length,
                                                 num_beams=num_beams, eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty, **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


# 关键：这个函数就是您在 main.py 里调用的函数
def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print(">>> 成功加载 BLIP 微调权重")
    return model


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    # 丢弃原权重中不需要的预训练头部
    #state_dict = {k: v for k, v in state_dict.items() if 'text_decoder.cls.predictions' not in k}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Missing keys (通常是您新加的模块参数):', msg.missing_keys)
    return model, msg


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")