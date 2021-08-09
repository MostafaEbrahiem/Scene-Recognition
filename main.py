
from CI_project import cnn as c
from CI_project import B_model as B
import numpy as np
def main():
    TRAIN_DIR,TEST_DIR=c.start()
    X_train,y_train,test=c.read(TRAIN_DIR,TEST_DIR,50)

    #vgg_model,history=c.transfer_learn(X_train,y_train,100)


    #data_gen=c.data_augmentation(X_train)
    #print(np.shape(X_train))
    #print(data_gen)
    #temp_model=B.pure_cnn_model(X_train,y_train,50)

    model=c.built_model(X_train,y_train,50,'CI_Project_cnn',0.001)
    #c.plot(history)
    c.test_res(test,model,50,TEST_DIR)

if __name__ == '__main__':
    main()