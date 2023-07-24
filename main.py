from functions import train_model
import yaml
from tensorflow.keras.models import save_model

with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)
    
hist, model = train_model(**yml)

# get maximum validation accuracy
if yml['multi_output']:
    acc = round(max(hist['val_avg_acc']) * 100, 1)
else:
    acc = round(max(hist['val_acc']) * 100, 1)

print('Best accuracy: ', acc)
save_ = input("Save model? (y/n)")
if save_ == 'y':
    save_model(model, f"{yml['model']}_model_{round(acc)}.h5")
    print('Saved')
