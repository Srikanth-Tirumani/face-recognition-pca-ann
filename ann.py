from sklearn.neural_network import MLPClassifier

def train_ann(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500
    )
    model.fit(X_train, y_train)
    return model
