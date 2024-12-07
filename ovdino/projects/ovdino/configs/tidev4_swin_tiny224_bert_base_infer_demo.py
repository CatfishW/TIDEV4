import os
import os.path as osp

from detrex.config import get_config

from .models.tidev4_swin_tiny224_bert_base import model

# model_root = os.getenv("MODEL_ROOT", "./inits")
# init_checkpoint = osp.join(model_root, "./swin", "swin_tiny_patch4_window7_224.pth")
#init_checkpoint = '/root/workspace/ZladWu/OV-DINO/ovdino/ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth'
init_checkpoint ='/root/workspace/ZladWu/OV-DINO/ovdino/wkdrs/ovdino_swin_tiny224_bert_base_24ep_text_visual_hybrid_openimages50K/model_0024999.pth'
# get default config
dataloader = get_config("common/data/custom_ovd.py").dataloader#get_config("common/data/coco_ovd.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = init_checkpoint
train.output_dir = "./wkdrs/tidev4_swin_tiny224_bert_base_24ep_text_visual_hybrid_openimages50K"

# max training iterations
train.max_iter = 200000
train.eval_period = 200000
train.log_period = 50
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
train.find_unused_parameters = True
model.device = train.device
model.inference_template = "full"
model.num_classes = 599


# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: (
    0.1 if "backbone" in module_name else 1
)

# modify dataloader config
dataloader.train.num_workers = 2

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 18

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
