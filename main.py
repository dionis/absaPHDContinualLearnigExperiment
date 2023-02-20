# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import DatasetItem

# now the dataset is a DatasetItem object, which has a name and a list of subdatasets
# e.g., SemEval dataset contains Laptop14, Restaurant14, Restaurant16 datasets
#from pyabsa import download_all_available_datasets


# config = ATEPC.ATEPCConfigManager.get_atepc_config_glove()  # get pre-defined configuration for GloVe model, the default embed_dim=300
config = (
    ATEPC.ATEPCConfigManager.get_atepc_config_english()
)  # this config contains 'pretrained_bert', it is based on pretrained models
dataset = ATEPC.ATEPCDatasetList.Restaurant16
#download_all_available_datasets()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

my_dataset = DatasetItem("my_dataset", ["my_dataset1", "my_dataset2"])
# my_dataset1 and my_dataset2 are the dataset folders. In there folders, the train dataset is necessary

from pyabsa import ModelSaveOption, DeviceTypeOption

config.batch_size = 8
trainer = ATEPC.ATEPCTrainer(
    config=config,
    dataset=dataset,
    from_checkpoint="english",
    # if you want to resume training from our pretrained checkpoints, you can pass the checkpoint name here
    auto_device=DeviceTypeOption.AUTO,
    path_to_save=None,  # set a path to save checkpoints, if it is None, save checkpoints at 'checkpoints' folder
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    load_aug=False,
    # there are some augmentation dataset for integrated datasets, you use them by setting load_aug=True to improve performance
)
