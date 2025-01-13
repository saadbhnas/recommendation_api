from recommendation_model.config.core import config
from recommendation_model.preprocess.pipline import recommendation_pipe
from recommendation_model.preprocess.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test = train_test_split(
        data[config.model_config.movies_metadata_features],  # predictors
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
   

    # fit model
    model_output = recommendation_pipe.fit(X_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=model_output)


if __name__ == "__main__":
    run_training()