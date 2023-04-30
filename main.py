if __name__ == '__main__':
    from functions import train_model
    import yaml
    import pickle
    
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
    
    hist, model = train_model(train_path, val_path,
                epochs=epochs,
                multi_output=multi_output,
                batch_size=batch_size,
		    blocks=blocks,
                patience=patience,
                dropout=dropout,
                augmentation=augmentation,
                verbose=verbose)

    if multi_output:
        score = 0
        for key, value in hist.items():
            if ('val' in key) and ('acc' in key):
                score += value[-1]
        print("\nAverage Accuracy: {:.2f}%\n".format((score*100) / 5))

    save_model = input("Save model? (y/n)")
    if save_model == 'y':
        pickle.dump(model, open('model.pkl', 'wb'))
        print('Saved')
