from urnai.trainers.filetrainer import FileTrainer

def test_solve_cartpole():
    trainer = FileTrainer('solve_cartpole_v1.json')
    trainer.start_training()
    assert trainer.logger.ep_avg_rewards[-1] >= 500
