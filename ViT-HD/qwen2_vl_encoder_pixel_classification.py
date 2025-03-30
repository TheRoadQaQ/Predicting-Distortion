from transformers.modeling_outputs import ImageClassifierOutput
from qwen2_vl_encoder import Qwen2VLVisionEncoder, Qwen2VLVisionConfig
from safetensors.torch import save_file, load_file
import torch.nn.functional as F
import torch
import torch.nn as nn

class Qwen2VLTForMalformPixelDetection(nn.Module):
    def __init__(self, num_classes, model_path,  use_focal_loss = False, focal_alpha = 1.0, focal_gamma = 0.0):
        super(Qwen2VLTForMalformPixelDetection, self).__init__()

        config = Qwen2VLVisionConfig.from_pretrained(model_path)
        self.vision_encoder = Qwen2VLVisionEncoder(config)
        #state_dict = load_file('./qwen-vl-encoder.safetensors')
        #self.vision_encoder.load_state_dict(state_dict)
        print(self.vision_encoder)
        self.num_classes = num_classes
        self.patch_size = 14

        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Define an MLP for classification
        # h = 1280
        self.mlp = nn.Sequential(
            nn.Linear(4*1280, 1280),
            nn.GELU(),
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    # num_patch = w//p * h//p = 32*32 = 1024
    # images: (b, w//p * h//p, h')
    # grid_thws: (b, 3)
    # labels: (b, resolution, resolution)
    def forward(self, images, grid_thws, pixel_labels = None):
       
        batch_size = images.shape[0]
        
        # (b, w//p * h//p, h')  ->  [(b * num_patch, h) x 32]
        _, hidden_states_list = self.vision_encoder(images, grid_thws) 

        hidden_state_8 = hidden_states_list[7]   # 索引从0开始，第8层对应索引7
        hidden_state_16 = hidden_states_list[15] # 第16层对应索引15
        hidden_state_24 = hidden_states_list[23] # 第24层对应索引23
        hidden_state_32 = hidden_states_list[31] # 第32层对应索引31

        # 在最后一个维度上进行拼接  (b * num_patch, 4h)
        outputs = torch.cat((hidden_state_8, hidden_state_16, hidden_state_24, hidden_state_32), dim=-1)

        # Classification
        # (b * num_patch, 4h) -> (b * num_patch, self.num_classes)
        logits = self.mlp(outputs)
        logits = logits.view(batch_size, -1, logits.shape[-1]) # (b, num_patch,  self.num_classes)
        height = width = int((logits.shape[1]) ** 0.5)

        logits = logits.permute(0,2,1) # (b, self.num_classes, num_patch)
        logits = logits.view(batch_size, -1, height, width) # (b, self.num_classes, height, width)

        # (b, self.num_classes, resolution, resolution)
        logits = F.interpolate(logits, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        # (b x resolution x resolution, self.num_classes)
        logits = logits.permute(0, 2, 3, 1).view(-1, self.num_classes)
      
        loss = None
        if pixel_labels is not None:
            # pixel_labels: (b, resolution, resolution,) 
            pixel_labels = pixel_labels.view(-1)
            #loss = F.cross_entropy(logits, pixel_labels)

            if self.use_focal_loss:
                loss = focal_loss(logits, pixel_labels, self.focal_gamma, self.focal_alpha)
            else:
                loss = F.cross_entropy(logits, pixel_labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits.view(batch_size, height*self.patch_size, width*self.patch_size, -1),
            hidden_states=None,
            attentions=None,
        )