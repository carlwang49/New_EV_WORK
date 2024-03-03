import numpy as np
import pandas as pd
import torch
from torch import cuda, Tensor
from torch.optim import Adam

from CaseRecommender.caserec.recommenders.item_recommendation.bprmf import BprMF
from CaseRecommender.caserec.utils.split_database import SplitDatabase
from torchkge.torchkge.data_structures import KnowledgeGraph
from torchkge.torchkge.models import DistMultModel
from torchkge.torchkge.sampling import BernoulliNegativeSampler
from torchkge.torchkge.utils import MarginLoss, DataLoader

from tqdm.autonotebook import tqdm

BATCH_SIZE = 8
LEARNING_RATE = 0.0002
RANK_LENGTH = 20
EPOCH = 500
EMB_DIM = 60
MARGIN = 0.3

def preprocessing_data(split=False):
    '''
    read KGE data

    input: split (default: False)
    output: training data, testing data
    '''
    if split:
        # BPRMF charging data
        SplitDatabase(input_file="./Dataset/BPRMF_train_5.csv", dir_folds="./Dataset/", n_splits=10, sep_read=",").k_fold_cross_validation()

    # KGE item-graph
    df = pd.read_csv("./Dataset/train.txt", sep="\t", header=None, names=["from", "to", "rel"])
    kg = KnowledgeGraph(df)
    kg_train, kg_test = kg.split_kg(sizes=(len(df)-100, 100))

    return kg_train, kg_test


if __name__ == "__main__":

    # 第一次需要先切資料， split 參數要設定為 true
    kg_train, kg_test = preprocessing_data()

    # BPRMF
    BPRMF = BprMF(train_file="./Dataset/folds/0/train.dat",
                  test_file="./Dataset/folds/0/test.dat",
                  output_file="./Result/ev_cs_preference.csv",
                  batch_size=BATCH_SIZE,
                  learn_rate=LEARNING_RATE,
                  rank_length=RANK_LENGTH,
                  epochs=EPOCH)
    BPRMF.initialize()

    # KGE: DistMult
    KGE = DistMultModel(EMB_DIM, kg_train.n_ent, kg_train.n_rel)
    criterion = MarginLoss(MARGIN)

    if cuda.is_available():
        cuda.empty_cache()
        KGE.cuda()
        criterion.cuda()

    optimizer = Adam(KGE.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=BATCH_SIZE, use_cuda='all')

    iterator = tqdm(range(EPOCH), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = KGE(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        BPRMF.input_with_kg(kg_q=KGE.get_embeddings()[0].cpu().numpy())
        BPRMF.fit()
        KGE.set_embeddings(torch.from_numpy(BPRMF.output_with_kg()).cuda())

        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                running_loss / len(dataloader)))


    KGE.normalize_parameters()
    user_emb = BPRMF.output_result()

    ent_emb, rel_emb = KGE.get_embeddings()
    ent_emb = ent_emb.cpu().numpy()
    rel_emb = rel_emb.cpu().numpy()
    np.savetxt("./Result/ent_emb.csv", ent_emb, delimiter="\t")
    np.savetxt("./Result/rel_emb.csv", rel_emb, delimiter="\t")
    np.savetxt("./Result/user_emb.csv", user_emb, delimiter="\t")