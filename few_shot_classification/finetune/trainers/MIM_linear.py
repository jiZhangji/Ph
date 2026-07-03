import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from tqdm import tqdm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'MSTAR_SOC': 'a photo of {}.',
}

from trainers.mim_sar_encoder import SARPretrainClassifier

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames):
        super().__init__()
        checkpoint_path = os.environ.get(
            'MIM_CKPT',
            '../weights/SAR-JEPA/checkpoint-200.pth',
        )
        model = SARPretrainClassifier(
            num_classes=len(classnames),
            checkpoint_path=checkpoint_path,
            linear_probe=True,
        )
        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head,
        )
        self.image_encoder = model.cuda()


    def forward(self, image):
        # image = torch.concat([image, image, image], 1)
        image_features = self.image_encoder(image)

        return image_features


@TRAINER_REGISTRY.register()
class MIM_linear(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading MAE (backbone: {cfg.MODEL.BACKBONE.NAME})')

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames)

        print('Turning off gradients in both the image and the text encoder')
        # for name, param in self.model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            # load_pretrained_weights(self.model.image_encoder, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        # self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('clip', self.model.image_encoder, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        # loss = F.cross_entropy(output2, label) + self.model.criteria(output1, label)
        loss = F.cross_entropy(output, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
        #     curr_result = self.test(split="val")
        #     is_best = curr_result > self.best_result
        #     if is_best:
        #         self.best_result = curr_result
        #         self.save_model(
        #             self.epoch,
        #             self.output_dir,
        #             val_result=curr_result,
        #             model_name="model-best.pth.tar"
        #         )

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)
