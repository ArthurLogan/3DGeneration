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


if __name__ == '__main__':
    import torch
    from torch import optim
    from model import ShapeAutoEncoder

    device = torch.device('cuda:3')
    epoch = 1000
    path = "./scheduler.pt"

    def demo(scheduler: WarmUpScheduler):
        for key, val in scheduler.state_dict().items():
            if isinstance(val, torch.optim.lr_scheduler.CosineAnnealingLR):
                for k_, v_ in val.state_dict().items():
                    print(f"{key}/{k_}: {v_}")
            else:
                print(f"{key}: {val}")
        print()

    def save():
        net = ShapeAutoEncoder(features=128, channels=512, layers=8, reg=True).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-4)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epoch, eta_min=0, last_epoch=-1)
        warmUpScheduler = WarmUpScheduler(
            optimizer=optimizer, multiplier=2, warm_epoch=epoch // 10, after_scheduler=cosineScheduler)
        
        for _ in range(200):
            warmUpScheduler.step()
        
        demo(warmUpScheduler)
        torch.save(warmUpScheduler.state_dict(), path)

    def load():
        net = ShapeAutoEncoder(features=128, channels=512, layers=8, reg=True).to(device)
        o_ = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-4)
        c_ = optim.lr_scheduler.CosineAnnealingLR(optimizer=o_, T_max=epoch, eta_min=0, last_epoch=-1)
        w_ = WarmUpScheduler(optimizer=o_, multiplier=2, warm_epoch=epoch // 10, after_scheduler=c_)
        w_.load_state_dict(torch.load(path))
        demo(w_)

    save()
    load()
