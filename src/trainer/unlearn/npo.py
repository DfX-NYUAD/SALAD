from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        # print("DEBUG forget_inputs keys:", forget_inputs.keys())
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        
        retain_inputs = inputs["retain"]
        # print("DEBUG retain_inputs keys:", retain_inputs.keys())
        # 🛠️ Fix malformed batch: e.g., {0: {...}} -> {...}
        if list(retain_inputs.keys()) == [0] and isinstance(retain_inputs[0], dict):
            retain_inputs = retain_inputs[0]

        # ✅ Validate all required keys are there
        required_keys = ["input_ids", "attention_mask", "labels"]
        for key in required_keys:
            if key not in retain_inputs:
                raise ValueError(f"Missing '{key}' in retain_inputs. Got keys: {retain_inputs.keys()}")


        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
