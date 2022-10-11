import torch
from transformers import BertTokenizerFast, VisualBertForPreTraining
from vb_utils.modeling_frcnn import GeneralizedRCNN
from vb_utils.processing_image import Preprocess
from vb_utils import utils
from vb_utils.utils import Config

class VB_Model:
    def __init__(self, device='cuda'):
        self.unmask_model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device

        # visual feature extractor
        ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
        ttrids = utils.get_data(ATTR_URL)
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(device)
        self.image_preprocess = Preprocess(self.frcnn_cfg)
    
    def process_img(self, img):
        images, sizes, scales_yx = self.image_preprocess(img)
        output_dict = self.frcnn(
            images.to(self.device),
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        roi_feats = output_dict["roi_features"]
        box_feats = output_dict["normalized_boxes"]
        proposed_boxes = output_dict["boxes"]
        extended_feats = torch.cat([roi_feats,box_feats, box_feats[:,:,2:3] - box_feats[:,:,0:1], box_feats[:,:,3:4] - box_feats[:,:,1:2]], dim=-1)
        return (roi_feats, extended_feats)

    def unmask(self, text, img_dir):
        inputs = self.tokenizer(text, return_tensors="pt")
        visual_embeds = self.process_img(img_dir)[0]
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs_unmask = self.unmask_model(**inputs)
        return outputs_unmask


