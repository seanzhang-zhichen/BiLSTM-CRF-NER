from train.train import Train

if __name__ == "__main__":
        model_train = Train()
        model_train.train(use_pretrained_w2v=True,  model_type="bilstm-crf")
