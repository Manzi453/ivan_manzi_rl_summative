### A2C: 10 configurations (300,032 frames each, seed 0)

| Run | Hyperparameters | Train return | Delivered (train) | Delivered (held-out) | Delivered (stress) | Frames to converge |
|---|---|---|---|---|---|---|
| cfg01 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | +9.06 | 79% | 74% | 68% | 180,224 |
| cfg02 | lr=0.0003, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | -1.62 | 0% | 0% | 0% | never |
| cfg03 | lr=0.003, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | +4.19 | 51% | 58% | 53% | 270,080 |
| cfg04 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=128, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | +7.73 | 71% | 72% | 62% | 190,080 |
| cfg05 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=512, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | +7.22 | 69% | 73% | 56% | 280,064 |
| cfg06 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.05, critic_coef=0.5, hidden=[64, 64] | +7.50 | 63% | 55% | 42% | 220,160 |
| cfg07 **(best)** | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.0, critic_coef=0.5, hidden=[64, 64] | +9.31 | 77% | 72% | 51% | 180,224 |
| cfg08 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=1.0, hidden=[64, 64] | +1.14 | 26% | 27% | 41% | 210,176 |
| cfg09 | lr=0.001, gamma=0.98, lmbda=0.9, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64] | +7.20 | 68% | 63% | 54% | 250,112 |
| cfg10 | lr=0.001, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[128, 128] | +7.41 | 61% | 66% | 57% | 270,080 |
