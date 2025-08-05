import os 
import json


epoch = [2]
unlearn = ["GradAscent", "GradDiff", "DPO", "NPO", "RMU","SimNPO"]
top_p = [0.0, 0.25, 0.5, 0.75, 1.0]
temp = [0.2, 0.4, 0.6, 0.8, 1.0]
forget_name = "VerilogEval"
result_forget_prob_all = {}
result_forget_rouge_all = {}
result_mink_all = {}
result_mink_plus_all = {}
result_privleak_all = {}
for e in epoch:
    for u in unlearn:
        if u == "original":
            if e == 3:
                base_folder = f"saves/unlearn/RTL_{forget_name}_Unlearn_GradAscent_ep{e}"
            else:
                continue
        else:
            base_folder = f"saves/unlearn/RTL_{forget_name}_Unlearn_{u}_ep{e}"
        result_forget_prob = []
        result_forget_rouge = []
        result_mink = []
        result_mink_plus = []
        result_privleak = []
        for top in top_p:
            for t in temp:
                if u == "original":
                    result_folder = f"{base_folder}/eval_learn_top_p_{top}_temp_{t}"
                else:
                    result_folder = f"{base_folder}/eval_unlearn_top_p_{top}_temp_{t}"
                if not os.path.exists(os.path.join(result_folder, "TOFU_SUMMARY.json")):
                    print(f"Result folder {result_folder} does not exist.")
                    continue
                with open(os.path.join(result_folder, "TOFU_SUMMARY.json")) as f:
                    result_data = json.load(f)
                    result_forget_prob.append(result_data["forget_Q_A_Prob"])
                    result_forget_rouge.append(result_data["forget_Q_A_ROUGE"])
                    result_mink.append(result_data["mia_min_k"])
                    result_mink_plus.append(result_data["mia_min_k_plus_plus"])
                    result_privleak.append(result_data["privleak"])
                print(result_folder)
        result_forget_prob_all[u] = result_forget_prob
        result_forget_rouge_all[u] = result_forget_rouge
        result_mink_all[u] = result_mink
        result_mink_plus_all[u] = result_mink_plus
        result_privleak_all[u] = result_privleak
        # print("forget_prob: ", result_forget_prob)

with open(f"result_unlearning_{forget_name}_ep{epoch[0]}.json", "w") as f:
    json.dump({
        "forget_prob": result_forget_prob_all,
        "forget_rouge": result_forget_rouge_all,
        "mia_min_k": result_mink_all,
        "mia_min_k_plus_plus": result_mink_plus_all,
        "privleak": result_privleak_all
    }, f, indent=4)
        
