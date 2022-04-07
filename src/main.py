from train.train import Train
from tools.get_ner_level_acc import precision

if __name__ == "__main__":

        use_pretrained_w2v = False
        model_type = "bert-bilstm-crf"
        
        model_train = Train()
        model_train.train(use_pretrained_w2v=use_pretrained_w2v,  model_type=model_type)
        
        text = "张铁柱，大学本科，毕业于东华理工大学，汉族。"

        result = model_train.predict(text, use_pretrained_w2v, model_type)
        print(result[0])

        result_dic = model_train.get_ner_list_dic(text, use_pretrained_w2v, model_type)
        print(result_dic)

