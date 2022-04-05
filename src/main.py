
from train.train import Train
from tools.get_ner_level_acc import precision

if __name__ == "__main__":

        # use_pretrained_w2v = True
        # model_type = "bilstm"
        
        # model_train = Train()
        # model_train.train(use_pretrained_w2v=use_pretrained_w2v,  model_type=model_type)
        
        # text = "张铁柱，毕业于东华理工大学，汉族人，籍贯: 江西 "

        # # result = model_train.predict(text, use_pretrained_w2v, model_type)
        # # print(result[0])

        # result_dic = model_train.get_ner_list_dic(text, use_pretrained_w2v, model_type)
        # print(result_dic)

        pre_labels = [['B-NAME', 'M-NAME', 'E-NAME', 'O', 'O', 'S-NAME', 'O', 'B-RACE', 'E-RACE', 'O', 'B-PRO', 'E-PRO', 'O']]
        true_labels = [['B-NAME', 'M-NAME', 'E-NAME', 'O', 'O', 'S-NAME', 'O', 'B-RACE', 'M-RACE', 'E-RACE', 'B-PRO', 'E-PRO', 'O']]
        res = precision(*pre_labels, true_labels)
        print(res)
