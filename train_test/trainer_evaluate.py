import json
import random

import autocuda
from metric_visualizer import MetricVisualizer
import sys
sys.path.append("..")
from pyabsa.framework.trainer_class.trainer_template import Trainer
from pyabsa.tasks.AspectPolarityClassification import APCTrainer
from pyabsa.tasks.AspectPolarityClassification import APCConfigManager
from pyabsa import ABSADatasetList
from pyabsa.tasks.AspectPolarityClassification import APCModelList
from pyabsa.framework.flag_class import TaskCodeOption
from pyabsa.framework.dataset_class import dataset_dict_class
import warnings
from pyabsa.utils.data_utils.dataset_item import DatasetItem
from pyabsa.tasks.AspectPolarityClassification.instructor.apc_instructor import APCTrainingInstructor
from pyabsa.utils.data_utils.dataset_manager import detect_dataset
from pyabsa.utils.logger.logger import get_logger
import os
from transformers import AutoConfig
from pyabsa.utils.pyabsa_utils import set_device

warnings.filterwarnings('ignore')


# seeds = [random.randint(0, 10000) for _ in range(3)]
seeds = [random.randint(0, 10000) for _ in range(1)]
device = autocuda.auto_cuda()

config1 = APCConfigManager.get_apc_config_english()
config1.model = APCModelList.BERT_MLP
config1.lcf = 'cdw'
config1.similarity_threshold = 1
config1.max_seq_len = 60
config1.dropout = 0.
config1.cache_dataset = False
config1.patience = 20
config1.optimizer = 'adamw'
# config1.pretrained_bert = 'microsoft/deberta-v3-large'
# config1.pretrained_bert = 'microsoft/deberta-v3-base'
# config1.pretrained_bert = 'yangheng/deberta-v3-large-absa'
config1.pretrained_bert = 'bert-base-uncased'
config1.num_epoch = 40
config1.log_step = 5
config1.SRD = 3
config1.eta = 1
config1.eta_lr = 0.001
config1.lsa = False
config1.learning_rate = 2e-5
config1.batch_size = 32
config1.evaluate_begin = 0
config1.l2reg = 1e-4
config1.seed = seeds[0]
config1.cross_validate_fold = -1  # disable cross_validate
config1.task_code = TaskCodeOption.Aspect_Polarity_Classification
save_path = '/home/dingyan/ABSA/PyABSA-LLM/experiments/laptop_aug_bert_base/checkpoints/bert_mlp_Laptop14_acc_92.67_f1_92.18'
dataset = DatasetItem("Laptop14", "111.arts_laptop14")
# dataset = DatasetItem("Laptop14", "113.Laptop14aug")
# dataset = ABSADatasetList.Laptop14
# dataset = ABSADatasetList.Restaurant14
# dataset = DatasetItem("Restaurant14", "114.Restaurant14aug")
# dataset=makedataset()
config1.MV = MetricVisualizer(
    config1.model.__name__ + '-' + dataset.dataset_name,
    trial_tag='Model & Dataset',
    trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# config1laptop = APCTrainer(
#     config=config1,
#     dataset=dataset,  # train set and test set will be automatically detected
#     checkpoint_save_mode=1,  # =None to avoid save model
#     auto_device=device,  # automatic choose CUDA or CPU
#     # load_aug=True,  # load augmented data
#     path_to_save=
#     "/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/checkpoints/bert_mlp_deberta_aug_base/laptop"
# )

config1.model_name = (
        config1.model.__name__.lower()
        if not isinstance(config1.model, list)
        else "ensemble_model"
    )
config1.logger = get_logger(os.getcwd(),
                           log_name=config1.model_name,
                           log_type="trainer")
config1.dataset = dataset
dataset_file = detect_dataset(
                config1.dataset,
                task_code=config1.task_code,
                load_aug=False,
                config=config1,
            )
config1.dataset_file = dataset_file
config1.dataset_name = config1.dataset.dataset_name
if config1.get("pretrained_bert", None):
    try:
        pretrain_config = AutoConfig.from_pretrained(config1.pretrained_bert)
        config1.hidden_dim = pretrain_config.hidden_size
        config1.embed_dim = pretrain_config.hidden_size
    except:
        pass
elif not config1.get("hidden_dim", None) or not config1.get("embed_dim", None):
    if config1.get("hidden_dim", None):
        config1.embed_dim = config1.hidden_dim
    elif config1.get("embed_dim", None):
        config1.hidden_dim = config1.embed_dim
    else:
        config1.hidden_dim = 768
        config1.embed_dim = 768

set_device(config1,device)

training_instructor = APCTrainingInstructor(config=config1)
training_instructor._reload_model_state_dict(save_path)
config1.input_cols=['text_indices']
training_instructor.model.to(device)
max_fold_acc, max_fold_f1 = training_instructor._evaluate_acc_f1(
    training_instructor.test_dataloader)
print(max_fold_acc)
print(max_fold_f1)