import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
    
#plotting graph
def plot_graph():
    X_set, Y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(Y_set)):
        plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j , 1], c = ListedColormap(('red', 'green'))(i), label = j, linewidths = 1, edgecolor = 'black')
    plt.title('Prediction boundary and training examples plotted')
    plt.legend()
    plt.show()
if __name__=="__main__":
    dataset1 = pd.read_csv('Social_Network_Ads.csv')
    x = dataset1.iloc[:, [2,3]].values
    y = dataset1.iloc[:, 4].values
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)
    while(True):        
        ch=int(input("MENU:\n1.KNN\n2.SVM\n3.Kernel SVM(rbf)\n4.Naive bayes\n5.Decision tree\n6.Random Forest \n7.Exit\nEnter Your Choice:"))
        if(ch==1):
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #n_neighbors parameter can be tuned, but will learn later.
        #minkowsky - 2 --> Euclidean distance
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()  
        elif ch==2:
            from sklearn.svm import SVC #Support vecture classifier
            classifier = SVC(kernel = 'linear', random_state = 0) #other options for kernel: 'poly', 'rbf' (gaussean), 'sigmoid', 'precomputed'
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()
            
        elif ch==3:
            from sklearn.svm import SVC
            classifier = SVC(kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()
        
        elif ch==4:
            from sklearn.naive_bayes import GaussianNB
            classifier = GaussianNB()
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()
        elif ch==5:
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()
        elif ch==6:
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, Y_train)
            y_pred = classifier.predict(X_test)
            plot_graph()
        elif ch==7:
            break
    
    print(accuracy_score(Y_test, y_pred))
    
