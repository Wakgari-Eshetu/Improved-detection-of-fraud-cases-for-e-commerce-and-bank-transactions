def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, batch_size=32, callbacks=None, model_path='model.h5'):
    trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, callbacks)
    loss, accuracy = evaluate_model(trained_model, X_test, y_test)
    save_model(trained_model, model_path)
    return loss, accuracy