### DQN: 10 configurations (300,000 frames each, seed 0)

| Run | Hyperparameters | Train return | Delivered (train) | Delivered (held-out) | Delivered (stress) | Frames to converge |
|---|---|---|---|---|---|---|
| cfg01 **(best)** | lr=0.001, gamma=0.99, double_dqn=False, eps_anneal_frames=20000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | +9.87 | 77% | 79% | 51% | 200,000 |
| cfg02 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | +7.82 | 69% | 86% | 66% | 270,016 |
| cfg03 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=50000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | +9.11 | 74% | 85% | 67% | 180,000 |
| cfg04 | lr=0.0003, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | -3.00 | 1% | 3% | 0% | never |
| cfg05 | lr=0.001, gamma=0.95, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | +8.54 | 66% | 74% | 65% | 230,016 |
| cfg06 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=hard, buffer_size=50000, batch_size=64, hidden=[128, 128] | -0.91 | 14% | 26% | 5% | never |
| cfg07 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=100000, batch_size=64, hidden=[128, 128] | -3.04 | 3% | 3% | 2% | 260,000 |
| cfg08 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=128, hidden=[128, 128] | +8.90 | 72% | 72% | 52% | 270,016 |
| cfg09 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.05, target=soft, buffer_size=50000, batch_size=64, hidden=[256, 256] | +7.21 | 67% | 83% | 48% | 230,016 |
| cfg10 | lr=0.001, gamma=0.99, double_dqn=True, eps_anneal_frames=100000, eps_end=0.1, target=soft, buffer_size=50000, batch_size=64, hidden=[128, 128] | +7.61 | 68% | 71% | 47% | 240,000 |
