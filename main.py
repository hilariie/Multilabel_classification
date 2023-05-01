if __name__ == '__main__':
    from functions import train_model
    import yaml
    from tensorflow.keras.models import save_model

    with open('config.yaml', 'r') as file:
        yml = yaml.safe_load(file)
    # Read parameters from yml file
    train_path = yml['train_path']
    val_path = yml['val_path']
    epochs = yml['epochs']
    multi_output = yml['multi_output']
    batch_size = yml['batch_size']
    blocks = yml['blocks']
    patience = yml['patience']
    dropout = yml['dropout']
    augmentation = yml['augmentation']
    verbose = yml['verbose']
    
    hist, model = train_model(train_path,
                              val_path,
                              epochs=epochs,
                              multi_output=multi_output,
                              batch_size=batch_size,
                              blocks=blocks,
                              patience=patience,
                              dropout=dropout,
                              augmentation=augmentation,
                              verbose=verbose)

    # get accuracies considering callback clause was called. This is done by default
    if multi_output:
        acc = round(max(hist['val_avg_acc']) * 100, 1)
    else:
        acc = round(max(hist['val_acc']) * 100, 1)

    print('Best accuracy: ', acc)
    save_ = input("Save model? (y/n)")
    if save_ == 'y':
        save_model(model, f'model_{round(acc)}.h5')
        print('Saved')

