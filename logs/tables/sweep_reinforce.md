### REINFORCE: 10 configurations (301,056 frames each, seed 0)

| Run | Hyperparameters | Train return | Delivered (train) | Delivered (held-out) | Delivered (stress) | Frames to converge |
|---|---|---|---|---|---|---|
| cfg01 | lr=0.001, gamma=0.99, baseline=batch_mean, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -2.49 | 0% | 0% | 0% | never |
| cfg02 | lr=0.001, gamma=0.99, baseline=value, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -3.23 | 0% | 0% | 0% | never |
| cfg03 | lr=0.0003, gamma=0.99, baseline=value, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -4.97 | 0% | 0% | 0% | never |
| cfg04 | lr=0.001, gamma=0.99, baseline=none, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -4.93 | 0% | 0% | 0% | never |
| cfg05 | lr=0.001, gamma=0.99, baseline=batch_mean, frames_per_batch=1024, entropy_coef=0.01, hidden=[64, 64] | -4.26 | 0% | 0% | 0% | never |
| cfg06 | lr=0.001, gamma=0.99, baseline=value, frames_per_batch=2048, entropy_coef=0.01, hidden=[64, 64] | -5.03 | 0% | 0% | 0% | never |
| cfg07 | lr=0.001, gamma=0.99, baseline=batch_mean, frames_per_batch=512, entropy_coef=0.05, hidden=[64, 64] | -2.89 | 0% | 0% | 0% | never |
| cfg08 | lr=0.003, gamma=0.99, baseline=batch_mean, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -2.86 | 0% | 0% | 0% | never |
| cfg09 **(best)** | lr=0.001, gamma=0.98, baseline=value, frames_per_batch=512, entropy_coef=0.01, hidden=[64, 64] | -1.81 | 0% | 0% | 0% | never |
| cfg10 | lr=0.0005, gamma=0.99, baseline=value, frames_per_batch=2048, entropy_coef=0.02, hidden=[64, 64] | -4.94 | 0% | 0% | 0% | never |
