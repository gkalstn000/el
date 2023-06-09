import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Scores() :
    def __call__(self, pred, true, type):
        '''
        :param pred: list type
        :param true: list type
        :param type: [train | test] string type
        :return: score dict
        '''
        accuracy = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        f1 = f1_score(true, pred)

        score_dict = {f'Acc/{type}' : torch.tensor(accuracy),
                      f'Recall/{type}': torch.tensor(recall),
                      f'Precision/{type}': torch.tensor(precision),
                      f'F1/{type}': torch.tensor(f1)}

        return score_dict
