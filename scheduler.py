from torch.optim.lr_scheduler import _LRScheduler


# warm up scheduler
# https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Scheduler.py
class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        """linear warm up scheduler"""
        self.mutiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)
    
    def get_lr(self):
        """linear interpolate -> after scheluder(cosine annealing etc)"""
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.mutiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.mutiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.mutiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    
    def step(self, epoch=None, metrics=None):
        """different strategy to update learning rate"""
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)


if __name__ == "__main__":
    import torch
    from torch import optim
    from model import ShapeAutoEncoder
    model = ShapeAutoEncoder(128, 512, 8, True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=0.0001)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1)
    warmUpScheduler = WarmUpScheduler(optimizer, multiplier=2.0, warm_epoch=2, after_scheduler=cosineScheduler)

    print(warmUpScheduler.base_lrs)

    for i in range(5):
        print(warmUpScheduler.get_lr())
        optimizer.step()
        warmUpScheduler.step()
