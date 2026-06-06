import matplotlib.pyplot as plt
plt.switch_backend('agg')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        
    def early_stop_conv(self, validation_loss, training_loss):
        if validation_loss < training_loss:
            self.counter = 0
        elif validation_loss > (training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def plotLosses(trainloss,valloss,epochs,Loss="Model loss",save_path=""):
    plt.plot(epochs,trainloss,'b',label="Train Loss")
    plt.plot(epochs,valloss,'r',label="Val Loss")
    plt.title(Loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path+Loss.split()[0]+"_lossplot.png")
    plt.close("all")
    return