
from train.train import Train

if __name__ == "__main__":

        
        model_train = Train()
        # model_train.train(use_pretrained_w2v=True,  model_type="bilstm-crf")
        
        text = "张铁柱，毕业于东华理工大学，汉族人，籍贯"

        result = model_train.predict(text)
        print(result[0])