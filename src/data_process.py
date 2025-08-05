import sys
import json
# this code is used to update the dataset based on the forget split and retain split
def update_dataset(forget_split, retain_split, holdout_split):
    base_path = "RTL_dataset/vericontaminated"
    if forget_split is None or retain_split is None:
        raise ValueError("forget_split and retain_split cannot be None")
    
    if forget_split == "RTL_Leaky":
        source_path = "RTL_dataset/saved/verileaky/dfx_IP_design_dataset.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["Instruction"]
            item_temp["Response"] = item["Response"][0]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_forget.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    elif forget_split == "RTL_Contamin":
        source_path = "RTL_dataset/saved/RTL-Repo/cropped_code_test_filtered_prompt.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["Instruction"]
            item_temp["Response"] = item["Response"]#[0]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_forget.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    elif forget_split == "RTL_Breaker":
        source_path = "RTL_dataset/saved/RTL-Breaker/posioned_all.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["Instruction"]
            item_temp["Response"] = item["Response"]#[0]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_forget.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    elif forget_split == "RTL_VerilogEval":
        source_path = "RTL_dataset/saved/vericontaminated/verilogeval.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["prompt"]
            item_temp["Response"] = item["ref"]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_forget.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    elif forget_split == "RTL_RTLLM":
        source_path = "RTL_dataset/saved/vericontaminated/rtllm.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["prompt"]
            item_temp["Response"] = item["ref"]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_forget.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError("forget_split is not defined")
    
    if retain_split == "RTL_Coder":
        source_path = "RTL_dataset/saved/RTL_Coder/RTL_Coder_ori.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["Instruction"]
            item_temp["Response"] = item["Response"][0]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_retain.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError("retain_split is not defined")
    
    if holdout_split == "VerilogEval":
        source_path = "RTL_dataset/saved/vericontaminated/verilogeval.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["prompt"]
            item_temp["Response"] = item["ref"]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_holdout.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")
    elif holdout_split == "RTLLM":
        source_path = "RTL_dataset/saved/vericontaminated/rtllm.json"
        with open(source_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        json_data = []
        for i, item in enumerate(data):
            item_temp = {}
            item_temp["Instruction"] = item["prompt"]
            item_temp["Response"] = item["ref"]
            # item_temp["task_id"] = f"taskid_{i}"
            json_data.append(item_temp)
        with open(f"{base_path}/test_holdout.json", "w") as f:
            for item in json_data:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <retain_split> <forget_split> <holdout_split>")
        sys.exit(1)

    retain_split = sys.argv[1]
    forget_split = sys.argv[2]
    holdout_split = sys.argv[3]

    update_dataset(forget_split, retain_split, holdout_split)
    print(f"Updated dataset with retain_split: {retain_split}, forget_split: {forget_split}, holdout_split: {holdout_split}")
    
