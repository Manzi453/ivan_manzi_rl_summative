### PPO: 10 configurations (301,056 frames each, seed 0)

| Run | Hyperparameters | Train return | Delivered (train) | Delivered (held-out) | Delivered (stress) | Frames to converge |
|---|---|---|---|---|---|---|
| cfg01 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +10.77 | 85% | 85% | 78% | 140,288 |
| cfg02 | lr=0.001, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +11.00 | 86% | 92% | 78% | 80,896 |
| cfg03 | lr=0.0001, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +8.66 | 77% | 76% | 59% | 300,032 |
| cfg04 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.1, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +10.57 | 85% | 84% | 77% | 130,048 |
| cfg05 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.3, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +10.21 | 83% | 86% | 75% | 160,768 |
| cfg06 **(best)** | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=8, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +11.25 | 87% | 84% | 70% | 90,112 |
| cfg07 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.05, frames_per_batch=1024, hidden=[64, 64] | -2.74 | 0% | 1% | 0% | 200,704 |
| cfg08 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.0, frames_per_batch=1024, hidden=[64, 64] | +11.05 | 87% | 79% | 76% | 230,400 |
| cfg09 | lr=0.0003, gamma=0.98, lmbda=0.9, clip_epsilon=0.2, epochs=4, entropy_coef=0.01, frames_per_batch=1024, hidden=[64, 64] | +10.25 | 82% | 82% | 69% | 140,288 |
| cfg10 | lr=0.0003, gamma=0.99, lmbda=0.95, clip_epsilon=0.2, epochs=4, entropy_coef=0.01, frames_per_batch=2048, hidden=[64, 64] | +11.01 | 86% | 82% | 76% | 120,832 |
