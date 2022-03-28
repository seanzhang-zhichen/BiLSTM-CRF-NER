
from tools.get_ner_level_acc import precision

pre = [['B-NAME', 'E-NAME', 'S-NAME', 'O', 'O', 'O', 'S-TITLE', 'B-TITLE', 'M-TITLE', 'E-TITLE', 'O']]
true_label = [['B-NAME', 'E-NAME', 'S-NAME', 'O', 'O', 'O', 'B-ORG', 'M-ORG', 'M-ORG', 'E-ORG', 'O']]
# x = x.split()

tags = [("B-NAME","M-NAME", "E-NAME"),
        ("B-TITLE","M-TITLE", "E-TITLE"), 
        ("B-ORG","M-ORG", "E-ORG"), 
        ("B-RACE","M-RACE", "E-RACE"), 
        ("B-EDU","M-EDU", "E-EDU"), 
        ("B-CONT","M-CONT", "E-CONT"),
        ("B-LOC","M-LOC", "E-LOC"), 
        ("B-PRO","M-PRO", "E-PRO")]


print(len(pre[0]))
print(len(true_label[0]))


result = precision(pre, true_label)
print(result)


from model.train import ModelTrain

model_train = ModelTrain()
model_train.train()
