from urnai.trainers.filetrainer import FileTrainer

JSON_FILE_PATH = \
    '/Users/marcocspc/Downloads/tmp/agent_drts_only_buildunits_god_save_us_all1/test.json'


def main():
    trainer = FileTrainer(JSON_FILE_PATH)
    trainer.trainings[0]['trainer']['params']['enable_save'] = False

    trainer.start_training(setup_only=True)

    trainer.agent.model.load(trainer.full_save_path)

    trainer.play()
    trainer.logger.save(trainer.full_save_play_path)


main()
