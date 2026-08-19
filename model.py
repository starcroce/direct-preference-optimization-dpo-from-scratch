"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    x_max = np.max(logits, axis=axis, keepdims=True)
    log_sum = np.log(
        np.sum(
            np.exp(logits - x_max),
            axis=axis, 
            keepdims=True,
        )
    )
    return logits - x_max - log_sum

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    x_max = np.max(logits, axis=axis, keepdims=True)
    e_x = np.exp(logits - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...
    token_ids = token_ids[:, :, np.newaxis]
    selected = np.take_along_axis(
        log_probs,
        token_ids,
        axis=-1,
    )
    return selected.squeeze(-1)

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    masked_logprobs = token_logprobs * mask
    return np.sum(masked_logprobs, axis=-1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # TODO: Initialize the policy language-model parameters with small random values
    if rng is None:
        rng = np.random.default_rng()

    embed = rng.normal(
        loc=0, 
        scale=0.02, 
        size=(vocab_size, d_model),
    )
    W_out = rng.normal(
        loc=0, 
        scale=0.02, 
        size=(d_model, vocab_size),
    )
    b_out = np.zeros(vocab_size)

    return {
        "embed": embed,
        "W_out": W_out,
        "b_out": b_out,
    }

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # TODO: Compute next-token logits for every position from policy params and token ids.
    embed, W_out, b_out = params["embed"], params["W_out"], params["b_out"]
    embs = embed[token_ids]
    outputs = embs @ W_out + b_out
    return outputs

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # TODO: Compute the total masked sequence log-probability under the current policy...
    logits = policy_token_logits(params, token_ids)
    log_probs = log_softmax(logits)
    token_logprobs = gather_token_logprobs(log_probs, token_ids)
    seq_logprobs = masked_sequence_logprob(token_logprobs, mask)

    return seq_logprobs

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # TODO: Compute gradients of the summed sequence log-probability w.r.t. params
    logits = policy_token_logits(params, token_ids)
    probs = softmax(logits, axis=-1)

    batch_size, seq_len = token_ids.shape
    d_logits = -probs
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    d_logits[batch_idx, seq_idx, token_ids] += 1
    d_logits *= mask[..., None]

    embed, W_out, b_out = params["embed"], params["W_out"], params["b_out"]
    hidden = embed[token_ids]
    d_b = np.sum(d_logits, axis=(0, 1))
    d_W = np.einsum("btd,btv->dv", hidden, d_logits)
    d_hidden = d_logits @ W_out.T
    d_embed = np.zeros_like(embed)
    np.add.at(d_embed, token_ids, d_hidden)

    return {
        "embed": d_embed,
        "W_out": d_W,
        "b_out": d_b,
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    # TODO: Compute the mean Bradley-Terry pairwise preference loss...
    margin = reward_chosen - reward_rejected
    softplus_neg_margin = np.logaddexp(0, -margin)
    loss = float(np.mean(softplus_neg_margin))
    return loss

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # TODO: Fraction of pairs where chosen reward is strictly higher than rejected.
    comp = reward_chosen > reward_rejected
    acc = np.mean(comp)
    return float(acc)

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # TODO: Package raw arrays into a list of preference-pair dictionaries
    res = []
    for i in range(len(prompts)):
        item = {
            "prompt": prompts[i],
            "chosen_ids": chosen_ids[i],
            "rejected_ids": rejected_ids[i],
            "chosen_mask": chosen_mask[i],
            "rejected_mask": rejected_mask[i],
        }
        res.append(item)
    return res

# Step 12 - sample_preference_batch
def sample_preference_batch(pairs, batch_size, rng=None):
    # TODO: Sample a mini-batch of preference pairs for one training step.
    if rng is None:
        rng = np.random.default_rng()
    
    idx_list = rng.choice(
        len(pairs), 
        size=batch_size,
        replace=batch_size > len(pairs),
    )
    
    res = {
        "chosen_ids": np.stack(
            [pairs[i]["chosen_ids"] for i in idx_list]
        ),
        "rejected_ids": np.stack(
            [pairs[i]["rejected_ids"] for i in idx_list]
        ),
        "chosen_mask": np.stack(
            [pairs[i]["chosen_mask"] for i in idx_list]
        ),
        "rejected_mask": np.stack(
            [pairs[i]["rejected_mask"] for i in idx_list]
        ),
    }

    if "prompt" in pairs[0]:
        res["prompt"] = np.stack(
            [pairs[i]["prompt"] for i in idx_list]
        )
    
    return res

# Step 13 - freeze_reference_logprobs
def freeze_reference_logprobs(ref_params, pairs):
    # TODO: Precompute and freeze reference-model sequence log-probabilities for every chosen and rejected response...
    res = []
    for p in pairs:
        chosen_ids = p["chosen_ids"]
        chosen_mask = p["chosen_mask"]
        rejected_ids = p["rejected_ids"]
        rejected_mask = p["rejected_mask"]
        
        chosen_seq_logprob = policy_sequence_logprob(
            ref_params, 
            chosen_ids[None, :], 
            chosen_mask[None, :],
        )
        rejected_seq_logprob = policy_sequence_logprob(
            ref_params, 
            rejected_ids[None, :], 
            rejected_mask[None, :],
        )
        
        item = {
            "chosen": float(chosen_seq_logprob.item()),
            "rejected": float(rejected_seq_logprob.item()),
        }
        res.append(item)
    return res

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # TODO: Compute the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return policy_logprob - reference_logprob

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Compute the scaled DPO pair margin for a batch of preference pairs
    chosen_log_ratios = policy_reference_logratio(
        policy_logprob_chosen, ref_logprob_chosen,
    )
    rejected_log_ratios = policy_reference_logratio(
        policy_logprob_rejected, ref_logprob_rejected,
    )
    margin = beta * (chosen_log_ratios - rejected_log_ratios)
    return margin

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    margins = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )
    loss = np.mean(np.logaddexp(0, -margins))
    return float(loss.item())

# Step 17 - dpo_loss_grad
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    # TODO: Evaluate DPO loss and return parameter gradients for the policy
    chosen_ids = batch["chosen_ids"]
    rejected_ids = batch["rejected_ids"]
    chosen_mask = batch["chosen_mask"]
    rejected_mask = batch["rejected_mask"]

    ref_logprob_chosen = ref_logprobs_batch["chosen"]
    ref_logprob_rejected = ref_logprobs_batch["rejected"]

    policy_logprob_chosen = policy_sequence_logprob(
        params, chosen_ids, chosen_mask,
    )
    policy_logprob_rejected = policy_sequence_logprob(
        params, rejected_ids, rejected_mask,
    )
    loss = dpo_loss(
        policy_logprob_chosen, 
        policy_logprob_rejected, 
        ref_logprob_chosen, 
        ref_logprob_rejected, 
        beta,
    )

    batch_size, seq_len = chosen_ids.shape
    margins = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )
    dpo_weights = (1 / (1 + np.exp(margins))) * beta / batch_size

    grads = {
        "W_out": np.zeros_like(params["W_out"]),
        "b_out": np.zeros_like(params["b_out"]),
        "embed": np.zeros_like(params["embed"]),
    }
    for i in range(batch_size):
        grad_chosen = sequence_logprob_grad(
            params, 
            chosen_ids[i : i + 1], 
            chosen_mask[i : i + 1],
        )
        grad_rejected = sequence_logprob_grad(
            params, 
            rejected_ids[i : i + 1], 
            rejected_mask[i : i + 1],
        )
        grads["W_out"] += dpo_weights[i] * (grad_rejected["W_out"] - grad_chosen["W_out"])
        grads["b_out"] += dpo_weights[i] * (grad_rejected["b_out"] - grad_chosen["b_out"])
        grads["embed"] += dpo_weights[i] * (grad_rejected["embed"] - grad_chosen["embed"])

    return loss, grads

# Step 18 - dpo_train_step
import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    # TODO: Execute one DPO gradient-descent update; return updated params + metrics
    loss, grads = dpo_loss_grad(
        params, 
        batch, 
        ref_logprobs_batch, 
        beta,
    )
    
    updated = {}
    for key in params:
        updated[key] = params[key] - learning_rate * grads[key]
    metrics = {"loss": loss}

    return updated, metrics

# Step 19 - train_dpo
def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    # TODO: Sample batches, run DPO train steps, record per-step metrics.
    if rng is None:
        rng = np.random.default_rng()

    history = []
    for step in range(num_steps):
        idx_list = rng.choice(
            len(pairs), 
            size=batch_size,
            replace=batch_size > len(pairs),
        )
        batch = {
            "chosen_ids": np.stack([
                pairs[i]["chosen_ids"] for i in idx_list
            ]),
            "rejected_ids": np.stack([
                pairs[i]["rejected_ids"] for i in idx_list
            ]),
            "chosen_mask": np.stack([
                pairs[i]["chosen_mask"] for i in idx_list
            ]),
            "rejected_mask": np.stack([
                pairs[i]["rejected_mask"] for i in idx_list
            ]),
        }

        if "prompt" in pairs[0]:
            batch["prompt"] = [
                pairs[i]["prompt"] for i in idx_list
            ]

        ref_logprobs_batch = {
            "chosen": ref_logprobs['chosen'][idx_list],
            'rejected': ref_logprobs['rejected'][idx_list],
        }
        params, metrics = dpo_train_step(
            params, 
            batch, 
            ref_logprobs_batch, 
            beta, 
            learning_rate,
        )
        
        item = {
            "step": step,
            "loss": metrics["loss"],
        }
        history.append(item)
    
    return params, history

# Step 20 - length_normalized_logprob
def length_normalized_logprob(seq_logprob, mask):
    # TODO: Normalize sequence log-probabilities by their valid token counts.
    length = np.sum(mask, axis=-1)
    return seq_logprob / length

# Step 21 - ipo_loss
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Evaluate mean squared IPO loss on unscaled log-ratio margins
    chosen_logratio = policy_reference_logratio(
        policy_logprob_chosen, ref_logprob_chosen,
    )
    rejected_logratio = policy_reference_logratio(
        policy_logprob_rejected, ref_logprob_rejected,
    )
    margins = chosen_logratio - rejected_logratio
    target = 1 / (2 * beta)
    loss = np.mean((margins - target) ** 2)
    return loss

# Step 22 - implicit_reward
def implicit_reward(policy_logprob, reference_logprob, beta):
    # TODO: return the vector of DPO implicit rewards for a batch of sequences
    logratio = policy_reference_logratio(
        policy_logprob, reference_logprob,
    )
    return beta * logratio

# Step 23 - preference_accuracy
def preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: fraction of pairs where chosen has higher implicit DPO reward
    chosen_rewards = implicit_reward(
        policy_logprob_chosen, 
        ref_logprob_chosen, 
        beta,
    )
    rejected_rewards = implicit_reward(
        policy_logprob_rejected,
        ref_logprob_rejected,
        beta,
    )
    acc = np.mean(chosen_rewards > rejected_rewards)
    return acc

# Step 24 - kl_to_reference
def kl_to_reference(policy_logprob, reference_logprob):
    # TODO: Estimate the mean KL divergence of the policy from the reference...
    logratio = policy_reference_logratio(
        policy_logprob, reference_logprob,
    )
    kl_divergence = np.mean(logratio)
    return float(kl_divergence)

# Step 25 - reward_margin_stats
def reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Summarize implicit-reward margins with mean, std, and frac positive.
    chosen_reward = implicit_reward(
        policy_logprob_chosen,
        ref_logprob_chosen,
        beta,
    )
    rejected_reward = implicit_reward(
        policy_logprob_rejected,
        ref_logprob_rejected,
        beta
    )
    margins = chosen_reward - rejected_reward
    
    mean_margin = np.mean(margins)
    std_margin = np.std(margins)
    frac_pos = np.mean(margins > 0)

    return {
        "mean_margin": mean_margin,
        "std_margin": std_margin,
        "frac_positive": frac_pos,
    }

# Step 26 - evaluate_dpo
def evaluate_dpo(params, pairs, ref_logprobs, beta):
    # TODO: Aggregate a full set of DPO evaluation metrics over a preference dataset.
    policy_logprob_chosen = []
    policy_logprob_rejected = []
    ref_logprob_chosen = []
    ref_logprob_rejected = []

    for pair, ref in zip(pairs, ref_logprobs):
        chosen_ids = pair["chosen_ids"]
        rejected_ids = pair["rejected_ids"]
        chosen_mask = pair["chosen_mask"]
        rejected_mask = pair["rejected_mask"]

        policy_logprob_chosen_curr = policy_sequence_logprob(
            params,
            chosen_ids[None, :],
            chosen_mask[None, :],
        )
        policy_logprob_rejected_curr = policy_sequence_logprob(
            params,
            rejected_ids[None, :],
            rejected_mask[None, :],
        )

        policy_logprob_chosen.append(policy_logprob_chosen_curr)
        policy_logprob_rejected.append(policy_logprob_rejected_curr)
        ref_logprob_chosen.append(ref["chosen"])
        ref_logprob_rejected.append(ref["rejected"])

    policy_logprob_chosen = np.asarray(
        policy_logprob_chosen
    ).reshape(-1, 1)
    policy_logprob_rejected = np.asarray(
        policy_logprob_rejected
    ).reshape(-1, 1)
    ref_logprob_chosen = np.asarray(
        ref_logprob_chosen
    ).reshape(-1, 1)
    ref_logprob_rejected = np.asarray(
        ref_logprob_rejected
    ).reshape(-1, 1)

    loss = dpo_loss(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )
    pref_acc = preference_accuracy(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )

    all_policy_logprobs = np.concatenate([
        policy_logprob_chosen,
        policy_logprob_rejected,
    ])
    all_ref_logprobs = np.concatenate([
        ref_logprob_chosen,
        ref_logprob_rejected,
    ])
    kl_to_ref = kl_to_reference(
        all_policy_logprobs,
        all_ref_logprobs,
    )

    margin_stats = reward_margin_stats(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )

    metrics = {
        "dpo_loss": loss,
        "preference_accuracy": pref_acc,
        "kl_to_reference": kl_to_ref,
        "mean_margin": margin_stats["mean_margin"],
        "std_margin": margin_stats["std_margin"],
        "frac_positive": margin_stats["frac_positive"],
    }
    return metrics

# Step 27 - run_dpo_pipeline
def run_dpo_pipeline(vocab_size, d_model, prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta, learning_rate, num_steps, batch_size, rng=None):
    # TODO: Wire the full DPO pipeline end-to-end from raw arrays to eval...
    if rng is None:
        rng = np.random.default_rng()

    params = init_policy_params(vocab_size, d_model, rng=rng)
    pairs = build_preference_pairs(
        prompts, 
        chosen_ids, 
        rejected_ids, 
        chosen_mask, 
        rejected_mask,
    )
    
    ref_logprobs_list = freeze_reference_logprobs(params, pairs)
    ref_logprobs_dict = {"chosen": [], "rejected": []}
    for item in ref_logprobs_list:
        ref_logprobs_dict["chosen"].append(item["chosen"])
        ref_logprobs_dict["rejected"].append(item["rejected"])
    
    ref_logprobs_dict["chosen"] = np.asarray(
        ref_logprobs_dict["chosen"]
    )
    ref_logprobs_dict["rejected"] = np.asarray(
        ref_logprobs_dict["rejected"]
    )

    params, history = train_dpo(
        params, 
        pairs,
        ref_logprobs_dict, 
        beta, 
        learning_rate, 
        num_steps, 
        batch_size, 
        rng=rng,
    )
    eval_metrics = evaluate_dpo(
        params, 
        pairs, 
        ref_logprobs_list, 
        beta,
    )

    res = {
        "params": params,
        "history": history,
        "eval_metrics": eval_metrics,
    }
    return res

