#import
import numpy as np

np.random.seed(1)
np.warnings.filterwarnings('ignore')

def read_data(filename):
    X = []
    Y = []
    with open(filename, "r") as data:
        for line in data:
            newline = line[:-1]
            currentline = newline.split(",")
            X.append(currentline[:-1])
            if currentline[-1]=='class-1':
                Y.append(1)
            if currentline[-1]=='class-2':
                Y.append(2)
            if currentline[-1]=='class-3':
                Y.append(3)
    X = [[float(y) for y in x] for x in X]
    X = np.array(X)
    Y = np.array(Y)

    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    return X,Y

def perceptron(X, y, epochs):
    n_samples, n_features = X.shape

    # init parameters
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for i, x_i in enumerate(X):
            output = np.dot(x_i, weights) + bias

            if output*y[i] <=0:
                weights += y[i] * x_i
                bias += y[i]

    return weights, bias
    
    
def predict(X, weights, bias):
    output = np.dot(X, weights) + bias
    y_predicted = np.where(output >= 0, 1, -1)
    return y_predicted

def prepdata2(x, y, c1, c2):
    a=0
    for i, y_i in enumerate(y): 
        if y_i not in [c1, c2]:
            x = np.delete(x, i-a, axis=0)
            y = np.delete(y, i-a)
            a=a+1

    for i, y_i in enumerate(y):
        if y_i == c1:
            y[i] = 1
        elif y_i == c2:
            y[i] = -1
    
    return x,y


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy*100

def test(X, Y, Xt, Yt):
    epochs = 20
    w, bias = perceptron(X,Y,epochs)

    predictions = predict(X, w, bias)
    print("Train Accuracy :", accuracy(Y, predictions),'%')

    predictions = predict(Xt, w, bias)
    print("Test Accuracy :", accuracy(Yt, predictions),'%')


def perceptron_reg(X, y, epochs, l):
    n_samples, n_features = X.shape

    # init parameters
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for i, x_i in enumerate(X):
            output = np.dot(x_i, weights) + bias

            if output*y[i] <=0:
                l1 = 1-(2*l)
                weights = (weights*l1)+ (y[i] * x_i)
                bias += y[i]

    return weights, bias

def test_reg(X, Y, Xt, Yt, l):
    epochs = 20

    w, b = perceptron_reg(X,Y,epochs, l)
    # print(w1)
    a = np.dot(X, w) + b
    at = np.dot(Xt, w) + b

    return a,at
    

def main():
    epochs = 20

    # read data
    traindata = "data/train.data"
    testdata = "data/test.data"
    
    X_train, Y_train = read_data(traindata)
    X_test, Y_test = read_data(testdata)

    # Data Transform
    # Class 1 & Class 2
    X12,Y12 = prepdata2(X_train, Y_train, c1=1, c2=2)
    X12t,Y12t = prepdata2(X_test, Y_test, c1=1, c2=2)
    
    # Class 2 & Class 3
    X23,Y23 = prepdata2(X_train, Y_train, c1=2, c2=3)
    X23t,Y23t = prepdata2(X_test, Y_test, c1=2, c2=3)
    
    # Class 1 & Class 3
    X13,Y13 = prepdata2(X_train, Y_train, c1=1, c2=3)
    X13t,Y13t = prepdata2(X_test, Y_test, c1=1, c2=3)

    #Class 1
    Y1 = np.array([1 if i == 1 else -1 for i in Y_train])
    Y1t = np.array([1 if i == 1 else -1 for i in Y_test])
    
    #Class 2
    Y2 = np.array([1 if i == 2 else -1 for i in Y_train])
    Y2t = np.array([1 if i == 2 else -1 for i in Y_test])

    #Class 3
    Y3 = np.array([1 if i == 3 else -1 for i in Y_train])
    Y3t = np.array([1 if i == 3 else -1 for i in Y_test])


    print('\nClass 1 vs Class 2')
    test(X12, Y12, X12t, Y12t)
    print('\nClass 2 vs Class 3')
    test(X23, Y23, X23t, Y23t)
    print('\nClass 1 vs Class 3')
    test(X13, Y13, X13t, Y13t)
    
    a1, a1t = test_reg(X_train, Y1, X_test, Y1t ,l = 0)
    a2, a2t = test_reg(X_train, Y2, X_test, Y2t ,l = 0)
    a3, a3t = test_reg(X_train, Y3, X_test, Y3t ,l = 0)

    a = np.empty((0,1),int)
    for i, x_i in enumerate(a1):
        a_i = np.argmax([a1[i],a2[i],a3[i]])+1
        a= np.append(a,a_i)
    print('\nTrain Accuracy for one vs rest :',accuracy(Y_train, a),'%')
    
    a = np.empty((0,1),int)
    for i, x_i in enumerate(a1t):
        a_i = np.argmax([a1t[i],a2t[i],a3t[i]])+1
        a= np.append(a,a_i)
    #print(a)
    print('Test Accuracy for one vs rest  :',accuracy(Y_test, a),'%')

    for l in [0.01, 0.1, 1.0, 10.0, 100.0]:
        a1, a1t = test_reg(X_train, Y1, X_test, Y1t ,l)
        a2, a2t = test_reg(X_train, Y2, X_test, Y2t ,l)
        a3, a3t = test_reg(X_train, Y3, X_test, Y3t ,l)

        a = np.empty((0,1),int)
        for i, x_i in enumerate(a1):
            a_i = np.argmax([a1[i],a2[i],a3[i]])+1
            a= np.append(a,a_i)
        # print(a)
        print('\nTrain Accuracy for one vs rest with k=',l,' :',accuracy(Y_train, a),'%')
    
        a = np.empty((0,1),int)
        for i, x_i in enumerate(a1t):
            a_i = np.argmax([a1t[i],a2t[i],a3t[i]])+1
            a= np.append(a,a_i)
        print('Test Accuracy for one vs rest with k=',l,'  :',accuracy(Y_test, a),'%')


if __name__ == "__main__":
    main()