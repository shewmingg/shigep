#coding:utf8
import torch as t
def celoss():
    return t.nn.CrossEntropyLoss()

def bloss():
    def loss(s,l):
        pass
    return loss


def topkloss(k=1):
    def loss(score, label):
        topk = score.data.topk(k=k)[1]
        result = (topk == (label.data.view(-1, 1).expand_as(topk)))
        result = result.sum(dim=1).float()
        mask = 1 - result.expand_as(score)
        mask = t.autograd.Variable(mask)
        score = mask * score
        return cel(score, label)

    return loss


# def reloss():
#     def loss(score,label):
#         mask = score.data.new(score.size()).fill_(0.8)
#         # mask.scatter_(1,t.arange(0,64).long().unsqueeze(1),1)
#         mask.scatter_(1,label.data.unsqueeze(1),1.4)
#         prob = t.nn.functional.softmax(score)
#         # mask[,label.data.long()] = 3
#         prob.data = (prob.data*mask).clamp(max=0.95)
#         # score = score*mask
#         return nll(prob,label)


#     return loss

def reloss():
    return ReLoss.apply


class ReLoss(t.autograd.Function):
    @staticmethod
    def forward(ctx, score, label, k=3):
        # import ipdb;ipdb.set_trace()
        prob = t.nn.functional.softmax(score).data

        topk = score.topk(k=k)[1]
        result = (topk == (label.view(-1, 1).expand_as(topk)))
        result = result.sum(dim=1, keepdim=True).float()
        mask = 1 - result.expand_as(score) * 0.5
        label_matrix = score.new(score.size()).fill_(0)
        label_matrix.scatter_(1, label.unsqueeze(1), 1)
        ctx.s = (mask, prob, label_matrix)
        return t.nn.functional.cross_entropy(score, label).data

    @staticmethod
    @t.autograd.function.once_differentiable
    def backward(ctx, gradoutput=None):
        mask, prob, label_matrix = ctx.s
        dscore = (prob - label_matrix) / prob.size(0)
        dscore = mask * dscore
        # import ipdb;ipdb.set_trace()
        return dscore, None, None

    @staticmethod
    def test(ctx):
        score = t.autograd.Variable(t.Tensor(3, 4).fill_(0), requires_grad=True)
        score.data[:, 0] = 1
        score.data[:, 1] = 2
        label = t.autograd.Variable(t.arange(0, 3)).long()
        b = ReLoss.apply(score, label, 1)
        b.backward()
        print score.grad
        mygrad = score.grad.data.clone()

        score.grad = None
        o = t.nn.functional.cross_entropy(score, label)
        o.backward()
        print score.grad


def l2loss():
    f = t.nn.MSELoss()

    def loss(score, label):
        mask = score.data.new(score.size()).fill_(0)
        mask.scatter_(1, label.data.unsqueeze(1), 1)
        prob = score
        # prob = t.nn.functional.sigmoid(score)
        return f(prob, t.autograd.Variable(mask))

    return loss
