from typing import Dict, Any, Union
from omegaconf import DictConfig

from data.qa import (
    QADataset,
    QAwithIdkDataset,
)
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    # ✅ Show which dataset this is
    # print("\n🔍 DEBUG _load_single_dataset")
    # print(f"📛 Dataset name: {dataset_name}")
    # print(f"🛠 Handler: {dataset_handler_name}")
    # print(f"📦 Dataset args:")
    # for k, v in dataset_args.items():
    #     print(f"   {k}: {v}")
    # print("=====================================\n")
    return dataset_handler(**dataset_args, **kwargs)


# def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
#     dataset = {}
#     for dataset_name, dataset_cfg in dataset_cfgs.items():
#         access_name = dataset_cfg.get("access_key", dataset_name)
#         dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
#     if len(dataset) == 1:
#         # return a single dataset
#         return list(dataset.values())[0]
#     # return mapping to multiple datasets
#     return dataset

def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    import traceback

    # print("\n🔍 DEBUG get_datasets")
    # print(f"🧩 Received dataset_cfgs keys: {list(dataset_cfgs.keys())}")

    dataset = {}

    for dataset_name, dataset_cfg in dataset_cfgs.items():
        # print(f"\n➡️ Attempting to load dataset: {dataset_name}")
        # print(f"📦 Raw config: {dataset_cfg}")
        access_name = dataset_cfg.get("access_key", dataset_name)

        try:
            dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
        except Exception as e:
            # print(f"❌ Failed to load dataset: {dataset_name}")
            # print(f"🚨 Exception: {e}")
            traceback.print_exc(limit=3)
            raise e  # re-raise so you still get the main stacktrace

    if len(dataset) == 1:
        # print("✅ Returning single dataset instance.")
        return list(dataset.values())[0]

    # print("✅ Returning multiple dataset instances.")
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)
    return data

# def get_data(data_cfg: DictConfig, mode="train", **kwargs):
#     import traceback
#     data = {}
#     data_cfg = dict(data_cfg)
#     anchor = data_cfg.pop("anchor", "forget")

#     print("\n📦 Starting get_data()")
#     print(f"📌 Mode: {mode}")
#     print(f"🎯 Anchor: {anchor}")
#     print("🧩 Splits to load:", list(data_cfg.keys()))

#     for split, dataset_cfgs in data_cfg.items():
#         print(f"\n🔄 Loading split: {split}")
#         print(f"📂 dataset_cfgs = {dataset_cfgs}")
#         traceback.print_stack(limit=3)  # Show where this split is being loaded

#         data[split] = get_datasets(dataset_cfgs, **kwargs)

#     if mode == "train":
#         return data

#     elif mode == "unlearn":
#         print("\n🔧 Entering unlearn mode")
#         unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
#         print("🧹 Unlearn splits:", list(unlearn_splits.keys()))

#         unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
#         data["train"] = unlearn_dataset
#         for split in unlearn_splits:
#             data.pop(split)

#     return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
