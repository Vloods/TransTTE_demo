
from fairseq.dataclass.configs import FairseqDataclass
from torch.nn import functional
import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
import time
import os.path as osp
import dgl

from torch_geometric.utils import add_self_loops, negative_sampling


def mape_loss(output, target, reduction = 'mean'):
    if reduction == 'mean':
        loss_mean = torch.mean((output - target).abs()/(target.abs() + 1e-8))
        return loss_mean
    else:
        loss_sum = torch.sum((output - target).abs()/(target.abs() + 1e-8))
        return loss_sum


@register_criterion("l1_loss_link_prediction", dataclass=FairseqDataclass)
class GraphPredictionL1LinkLoss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True, batched_data = None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 1:, :]
        
        # print('sample logits',logits.size())
        # print('sample x',sample["net_input"]["batched_data"]["x"].size())
        # print('sample edge_label_index_pos',sample["net_input"]["batched_data"]["edge_label_index_pos"].size())
        # print('sample edge_label_index_neg',sample["net_input"]["batched_data"]["edge_label_index_neg"].size())
        # print('sample edge_label_pos',sample["net_input"]["batched_data"]["edge_label_pos"].size())
        # print('sample edge_label_neg',sample["net_input"]["batched_data"]["edge_label_neg"].size())
        
        edge_label_index = torch.cat((sample["net_input"]["batched_data"]["edge_label_index_pos"] , sample["net_input"]["batched_data"]["edge_label_index_neg"]), dim = 1)
        edge_label = torch.cat((sample["net_input"]["batched_data"]["edge_label_pos"] , sample["net_input"]["batched_data"]["edge_label_neg"]))
        out = (logits[:,edge_label_index[0],:] * logits[:,edge_label_index[1],:]).sum(dim=-1).view(-1)      
        
        # print('sample edge_label_index',edge_label_index.size())
        # print('sample edge_label',edge_label.size())
        # print('sample out',out.size())
        
        # logits = logits.to('cpu')
        # print('sample logits r',logits[0, 0, 0])
        # print('sample edge_label_index r',edge_label_index)
        # time.sleep(100)
        # loss = nn.BCEWithLogitsLoss(out, edge_label)
        loss = functional.binary_cross_entropy_with_logits(
            out.float(), edge_label.float(), reduction="sum"
        )
        print(loss)
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    
    
@register_criterion("mse_loss", dataclass=FairseqDataclass)
class GraphPredictionMSELoss(FairseqCriterion):

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        loss = nn.MSELoss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    
@register_criterion("rmse_loss", dataclass=FairseqDataclass)
class GraphPredictionRMSELoss(FairseqCriterion):

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        loss = torch.sqrt(nn.MSELoss(reduction="sum")(logits, targets[: logits.size(0)]))

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    
@register_criterion("mape_loss", dataclass=FairseqDataclass)
class GraphPredictionMAPELoss(FairseqCriterion):

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        loss_mean = mape_loss(logits, targets[: logits.size(0)], 'mean')
        loss_sum = mape_loss(logits, targets[: logits.size(0)], 'sum')
        print('mape mean', loss_mean)
        print('mape sum', loss_sum)
        time.sleep(100)
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True