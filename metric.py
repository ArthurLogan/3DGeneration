import torch


# metric calculator
class Metric:
    @classmethod
    def get(cls, out, gt, metrics):
        res = dict()
        for metric in metrics:
            if metric == 'iou':
                res[metric] = cls._calc_iou(out, gt)
            elif metric == 'pr':
                res[metric] = cls._calc_pr(out, gt)
            else:
                raise Exception(f'Unsupported metric: {metric}')
        return res
    
    @classmethod
    def _calc_iou(cls, out, gt):
        """[B, N] -> 1"""
        intersection = ((out + gt) == 2).int().sum(dim=1)
        union = ((out + gt) > 0).int().sum(dim=1)
        iou = intersection / union
        return iou.mean()

    @classmethod
    def _calc_pr(cls, out, gt):
        """[B, N] -> 1"""
        intersection = ((out + gt) == 2).int().sum(dim=1)
        precision = intersection / ((out == 1).int().sum(dim=1) + 1e-5)
        recall = intersection / ((gt == 1).int().sum(dim=1) + 1e-5)
        return precision.mean(), recall.mean()


if __name__ == '__main__':
    out = (torch.randn((2, 3)) > 0).int()
    gt = torch.stack([out[0], torch.randn(3) > 0], dim=0).int()
    metric_outs = Metric.get(out, gt, metrics=['iou', 'pr'])
    print(out)
    print(gt)
    print(metric_outs)
