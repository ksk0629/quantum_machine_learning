import torch
from tqdm.auto import tqdm


class TorchTrainer:
    """Trainer class."""

    def __init__(
        self,
        model: torch.nn.Module,
    ):
        """Initialise this trainer.

        :param nn.Module model: model to train
        """
        self.model = model
        self.current_epoch = 0

        # self.criterion = nn.NLLLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.model.parameters())

        self.train_loss_history = []
        self.test_loss_history = []

        self.train_accuracy_history = []
        self.test_accuracy_history = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __update(self, data: torch.Tensor, label: torch.Tensor) -> float:
        """Update the parameters of the model.

        :param torch.Tensor data: data for training
        :param torch.Tensor label: label for training
        :return float: loss value
        """
        # Initialise the gradients.
        self.optimiser.zero_grad()
        # Calculate the loss.
        loss = self.__calculate_loss(data=data, label=label)
        # Perform the backpropagation.
        loss.backward()
        # Update the parameters.
        self.optimiser.step()

        return loss.item()

    def __calculate_loss(
        self, data: torch.Tensor, label: torch.Tensor
    ) -> torch.nn.modules.loss._Loss:
        """Calculate the loss.

        :param torch.Tensor data: data for calculating loss
        :param torch.Tensor label: data for calculating loss
        :return nn._Loss: loss
        """
        # Classify the data.
        output = self.model(data)
        # Calculate the loss value and accumulate it.
        loss = self.criterion(output, label)

        return loss

    def __train_one_epoch(self, train_loader: torch.utils.data.DataLoader):
        """Train the model."""
        self.model.train()

        train_loss = 0
        with tqdm(train_loader) as tepoch:
            # Initialise the count of correctly predicted data.
            total_correct = 0
            total = 0

            for data, label in tepoch:
                # Set the description.
                tepoch.set_description(f"Epoch {self.current_epoch} (train)")

                # Transfer the data and label to the device.
                data, label = data.to(self.device), label.to(self.device)

                # Update the parameters.
                loss_value = self.__update(data=data, label=label)
                train_loss += loss_value

                # Get the number of correctly predicted ones.
                predicted_label = self.model.classify(data)
                num_correct = (predicted_label == label).sum().item()
                total_correct += num_correct
                total += len(label)

                # Set the current loss and accuracy.
                batch_accuracy = num_correct / len(label)
                tepoch.set_postfix(
                    {"Loss_train": loss_value, "Accuracy_train": batch_accuracy}
                )

        # Store the loss value.
        average_train_loss = train_loss / len(train_loader)
        self.train_loss_history.append(average_train_loss)

        # Store the accuracy.
        accuracy = total_correct / total
        self.train_accuracy_history.append(accuracy)

    def __eval_one_epoch(self, validation_loader: torch.utils.data.DataLoader):
        """Evaluate the model."""
        self.model.eval()

        test_loss = 0
        with tqdm(validation_loader) as tepoch:
            with torch.no_grad():  # without calculating the gradients.
                # Initialise the count of correctly predicted data.
                total_correct = 0
                total = 0

                for data, label in tepoch:
                    # Set the description.
                    tepoch.set_description(f"Epoch {self.current_epoch} (test)")

                    # Transfer the data and label to the device.
                    data, label = data.to(self.device), label.to(self.device)
                    # Calculate the loss.
                    loss_value = self.__calculate_loss(data=data, label=label).item()
                    test_loss += loss_value

                    # Get the number of correctly predicted ones.
                    predicted_label = self.model.classify(data)
                    num_correct = (predicted_label == label).sum().item()
                    total_correct += num_correct
                    total += len(label)

                    # Set the current loss and accuracy.
                    batch_accuracy = num_correct / len(label)
                    tepoch.set_postfix(
                        {
                            "Loss_test": loss_value,
                            "Accuracy_test": batch_accuracy,
                        }
                    )
        # Store the loss value.
        average_test_loss = test_loss / len(validation_loader)
        self.test_loss_history.append(average_test_loss)

        # Store the accuracy.
        accuracy = total_correct / total
        self.test_accuracy_history.append(accuracy)

    def train_and_eval(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        epochs: int,
    ):
        """Train and evaluate the model self.epochs times."""
        for current_epoch in range(1, epochs + 1):
            self.current_epoch = current_epoch
            self.__train_one_epoch(train_loader=train_loader)
            self.__eval_one_epoch(validation_loader=validation_loader)
