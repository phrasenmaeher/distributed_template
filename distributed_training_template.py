import tensorflow as tf
import argparse

# Todo: modify or add more models (e.g., discriminator and generator in a GAN setting) and optimizers
model: tf.keras.Model
optimizer: tf.keras.optimizers.Optimizer

# Todo: modify or add further train losses and train metrics
train_loss_object1: tf.losses.Loss
train_metric1: tf.metrics.Metric

# Todo: modify and add test losses or test metrics
# ...

# Todo: modify and add validation losses or validation metrics
# ...

GLOBAL_BATCH_SIZE: int

strategy: tf.distribute.Strategy


@tf.function
def train_step(inputs):
    """
    Applies a single batch update to the models
    :param inputs: The input data
    :return: the loss(es)
    """
    # Todo: Replace with correct unpacking
    features, labels = inputs  # unpack the tuple

    with tf.GradientTape(persistent=True) as tape:
        # Todo: Replace with gradient calculation here
        model_output = model(features, training=True)
        # Todo: calculate loss here (see compute loss down below)
        loss = compute_loss(ground_truth=labels, predictions=model_output)

    # Todo: apply gradients via optimizers
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Todo: update metrics
    train_metric1.update_state(y_true=labels, y_pred=model_output)
    return loss


@tf.function
def distributed_train_step(dataset_inputs):
    # distribute the computation and get per-replica (i.e., per device) losses
    # Todo: modify for task at hand (e.g., mutliple losses, inputs)
    loss = strategy.run(train_step, args=(dataset_inputs,))

    # Todo: aggregate train loss (see below for example)
    # scale the (individual) losses
    scaled_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    return scaled_loss


# the test step can often also be used for validation during training, but a separate validation step is given below
@tf.function
def test_step(inputs):
    """
    Runs a single (batch) test step on the model(s) after training has finished
    :param inputs: The input data
    :return: The loss(es)
    """
    # Todo: Replace with correct unpacking
    features, labels = inputs

    # Todo: implement test logic here;
    # the main difference to training is that we skip calculating gradients and updating weights
    model_output = model(features, training=False)

    # Todo: replace with test loss
    loss = compute_loss(ground_truth=labels, predictions=model_output)

    # Todo: add and update any test metrics (separate from the train metrics!) if required (see train_step)

    return loss


@tf.function
def distributed_test_step(dataset_inputs):
    """
    This method takes a (split of a) dataset and call the test step to run a single test step
    :param dataset_inputs: A (split) of a dataset
    :return: The reduced loss(es)
    """
    # distribute the computation and get per-replica (i.e., per device) losses
    # Todo: modify test step for task at hand (e.g., mutliple losses, inputs)
    per_replica_test_loss = strategy.run(test_step, args=(dataset_inputs,))

    # Todo: aggregate test loss (see below for example)
    # scale the (individual) losses
    test_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_test_loss, axis=None)

    return test_loss


def validation_step(inputs):
    """
    Similar to the test step, but runs a single validation step during training
    :param inputs: The dataset input
    :return: The loss(es)
    """
    features, labels = inputs
    # Todo: implement validation logic here
    model_output = model(features, training=False)
    # Todo: replace with validation or test loss
    loss = compute_loss(ground_truth=labels, predictions=model_output)

    # Todo: add and upadte any validation metrics (separate from the train metrics -> see train step)
    return loss


@tf.function
def distributed_validation_step(dataset_inputs):
    """
    Similar to the distributed test step, but during training
    :param dataset_inputs: A (split) of a dataset
    :return: The reduced loss(es)
    """
    # Todo: modify validation step for task at hand (e.g., mutliple losses, inputs)
    per_replica_validation_loss = strategy.run(fn=validation_step, args=(dataset_inputs,))

    # Todo: aggregate validation loss (see below for example)
    validation_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_validation_loss, axis=None)

    return validation_loss


def compute_loss(ground_truth, predictions):
    """
    Compute the loss, given the ground truth and predicted values
    :param ground_truth:
    :param predictions:
    :return: scaled loss
    """

    # Todo: modify loss calculation for task at hand (e.g., multiple losses, custom losses)
    per_replica_loss = train_loss_object1(y_true=ground_truth, y_pred=predictions)

    # Todo: scale the loss
    # see alert here https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
    scaled_per_replica_loss = tf.nn.compute_average_loss(per_replica_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    return scaled_per_replica_loss


def create_loss_objects(strategy: tf.distribute.Strategy):
    """
    Creates the loss object without any default reduction
    :param strategy:
    :return: the loss object(s)
    """

    # Strategy's scope is used to make the loss distribution-ready
    with strategy.scope():
        # Todo: define the loss object(s) for task at hand
        # this is only a sample loss
        loss_obj1 = tf.keras.losses.Loss(reduction=tf.keras.losses.Reduction.NONE)

    return loss_obj1


def create_metrics(strategy: tf.distribute.Strategy):
    """
    Create the required metrics, make them distribution ready
    :param strategy:
    :return:
    """

    # Strategy's scope is used to make the metrics distribution-ready
    with strategy.scope():
        # Todo: define the metric(s) for training, validation and testing
        # Todo: give them custom, meaningful names to keep the overview
        train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")

    return train_accuracy


def create_model_and_optimizer(strategy: tf.distribute.Strategy):
    """
    Create the model and optimizer and make the distribution aware
    :param strategy:
    :return: model and
    """
    with strategy.scope():
        # Todo: modify model creation for task at hand
        # Todo: if required, define multiple models
        model = None

        # Todo: create optimizer(s)
        optimizer = None

    return model, optimizer


def train(args: dict, distributed_train_dataset, distributed_validation_dataset):
    """
    Trains the model on the distributed train dataset and validates it on the distributed validation dataset
    :param args: command line parameter
    :param distributed_train_dataset: train dataset
    :param distributed_validation_dataset: validation dataset
    :return: nothing
    """
    # Todo: modify for task at hand
    # below code is sample code only
    template = "Epoch: (), Train Loss: (), Test Loss: {}"
    for epoch in range(args['epochs']):

        # iterate over the training dataset
        total_train_loss = 0.0
        num_batches = 0
        for elem in distributed_train_dataset:
            total_train_loss += distributed_train_step(dataset_inputs=elem)
            num_batches += 1
        train_loss = total_train_loss / num_batches

        # iterate over the test dataset:
        total_test_loss = 0.0
        num_batches = 0
        for elem in distributed_validation_dataset:
            total_test_loss += distributed_validation_step(dataset_inputs=elem, strategy=strategy)
            num_batches += 1
        test_loss = total_test_loss / num_batches

        print(template.format(epoch + 1, train_loss, test_loss))

        # Todo: get and print results for all metrics
        # Todo: add any additional train/test metrics
        metric1_results = train_metric1.result().numpy()
        print(f"Results for metric1 are {metric1_results}")

        # Todo: reset all losses and all train/test metrics
        train_metric1.reset_state()
        train_loss_object1.reset_state()


def test(args: dict, distributed_test_dataset):
    """
    Test the trained model on the test dataset
    :param args: dictionary of any neccessary args
    :param distributed_test_dataset: the test dataset
    :return: nothing
    """
    # Todo: modify for task at hand
    total_test_loss = 0.0
    num_batches = 0
    for elem in distributed_test_dataset:
        total_test_loss += distributed_test_step(dataset_inputs=elem, strategy=strategy)
        num_batches += 1
    test_loss = total_test_loss / num_batches

    # Todo: print all test losses and test metrics
    print(f"Test loss is {test_loss}")


def main(args: dict):
    """
    The main driver function
    :param args: (commandline) arguments
    :return: nothing
    """

    # Todo: globaly register all objects that are used inside tf.function calls
    global strategy, train_loss_object1, train_metric1, model, optimizer, GLOBAL_BATCH_SIZE

    # Todo: 1. create strategy object for the task at hand
    # see here for an overview: https://www.tensorflow.org/api_docs/python/tf/distribute
    strategy: tf.distribute.Strategy = None
    GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * args['batch_size']

    # Todo: 2. create losses
    train_loss_object1 = create_loss_objects(strategy=strategy)

    # Todo: 3. create metrics
    train_metric1 = create_metrics(strategy=strategy)

    # Todo: 4. create model and optimizer
    model, optimizer = create_model_and_optimizer(strategy=strategy)

    # Todo: 5. load and distribute dataset
    # implement data loading in separate function
    distributed_train_dataset: tf.data.Dataset = None
    distributed_validation_dataset: tf.data.Dataset = None
    distributed_test_dataset: tf.data.Dataset = None

    # Todo: 6. train model
    train(args=args, distributed_train_dataset=distributed_train_dataset,
          distributed_validation_dataset=distributed_validation_dataset)

    # Todo 7. test trained model
    test(args=args, distributed_test_dataset=distributed_test_dataset)


if __name__ == '__main__':
    # Todo: if required add command line argument parser
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--epochs", type=int, default=1)
    argument_parser.add_argument("--batch_size", type=int, default=16)
    arguemnts = argument_parser.__dict__

    main(args=arguemnts)
