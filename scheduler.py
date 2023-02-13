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
