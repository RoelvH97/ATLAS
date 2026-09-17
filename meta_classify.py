import hydra

from meta_classification import ClassificationTrainer


@hydra.main(version_base=None, config_path="configs", config_name="classify")
def main(config):
    ClassificationTrainer(config).run()


if __name__ == "__main__":
    main()
