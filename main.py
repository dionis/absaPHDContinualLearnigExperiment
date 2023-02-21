# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pyabsa import AspectTermExtraction as ATEPC


# now the dataset is a DatasetItem object, which has a name and a list of subdatasets
# e.g., SemEval dataset contains Laptop14, Restaurant14, Restaurant16 datasets
#from pyabsa import download_all_available_datasets


# config = ATEPC.ATEPCConfigManager.get_atepc_config_glove()  # get pre-defined configuration for GloVe model, the default embed_dim=300


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# my_dataset1 and my_dataset2 are the dataset folders. In there folders, the train dataset is necessary

#from pyabsa import ModelSaveOption, DeviceTypeOption

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config = ATEPCConfigManager.get_atepc_config_english()

atepc_config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC
dataset_path = ABSADatasetList.Restaurant14
# or your local dataset: dataset_path = 'your local dataset path'

aspect_extractor = ATEPCTrainer(config=atepc_config,
                                dataset=dataset_path,
                                from_checkpoint='',  # set checkpoint to train on the checkpoint.
                                checkpoint_save_mode=1,
                                auto_device=True
                                ).load_trained_model()
