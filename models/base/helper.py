# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import util.lr_sched as lr_sched



def cross_entropy_with_label_smoothing(pred, target, smoothing=0.0):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()





def base_train(model, trainloader, optimizer, epoch, args):
    tl = Averager()
    ta = Averager()
    model = torch.compile(model)
    model = model.train(True)
    # standard classification for pretrain

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        lr_sched.adjust_learning_rate(optimizer, epoch / len(trainloader) + epoch, args)

        logits, embed = model(data)
        logits = logits[:, :args.base_class]
        # loss = F.cross_entropy(logits, train_label)
        loss = cross_entropy_with_label_smoothing(logits, train_label, smoothing=0.5)


        acc = count_acc(logits, train_label)

        total_loss = loss #+ loss_distill * (1 - epoch/args.epochs_base)

        lrc = optimizer.param_groups[0]['lr']
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits, _ = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)

            # if session == 8:
            #     draw(logits, test_label)

                

            acc = count_acc(logits, test_label)
            

            vl.add(loss.item())
            va.add(acc)
            

        vl = vl.item()
        va = va.item()
        torch.cuda.empty_cache()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va


def adjust_logits(logits, session):
    # 分割logits为两部分
    softmax = F.softmax(logits, dim=1)
    logits_front = logits[:, :60]  # 取前60维
    logits_back = logits[:, 60:]   # 取后session*5维

    front_sum = (torch.sum(logits_front, dim=1) - torch.max(logits_front, dim=1)[0])/(60-1)
    back_sum = (torch.sum(logits_back, dim=1) - torch.max(logits_back, dim=1)[0])/((session*5)-1)
    




    print(front_sum-back_sum)
    # 将缩放后的logits与原始的前60维logits取平均
    logits_adjusted = torch.cat((logits_front-front_sum.unsqueeze(dim=1)+back_sum.unsqueeze(dim=1), logits_back), dim=1)

    return logits_adjusted



def count(logits, label):
    probabilities = F.softmax(logits, dim=1)
    correct_probabilities = probabilities[torch.arange(probabilities.size(0)), label]

    with open('app.txt', 'a') as file:
        for prob in correct_probabilities:
            file.write(f'{prob.item()}\n')
    return 





def test_acc(logits, test_label):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == test_label).sum().item()
    total = test_label.size(0)
    accuracy = correct / total
    return accuracy