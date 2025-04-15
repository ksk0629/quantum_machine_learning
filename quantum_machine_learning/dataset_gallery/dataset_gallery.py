import numpy as np
from sklearn import datasets
import torchvision


class DatasetGallery:
    """Dataset Gallery class.
    For now, I will not create test for this class."""

    @staticmethod
    def get_dataset(
        name: str, dataset_options: None | dict = None
    ) -> tuple[np.ndarray, list[str]]:
        """Get dataset from dataset name.

        :param str name: dataset name
        :param None | dict dataset_options: dataset options, defaults to None
        :raises ValueError: if the given dataset name was not valid
        :return tuple[np.ndarray, list[str]]: dataset
        """
        # Selecet a dataset.
        match name:
            case "iris":
                get_dataset = DatasetGallery.get_iris
            case "wine":
                get_dataset = DatasetGallery.get_wine
            case "pm_numbers":
                get_dataset = DatasetGallery.get_pm_numbers
            case "ls_numbers":
                get_dataset = DatasetGallery.get_ls_numbers
            case "mnist":
                get_dataset = DatasetGallery.get_mnist
            case "get_hor_ver_images":
                get_dataset = DatasetGallery.get_hor_ver_images
            case _:
                name_error = f"The name must be in (iris, wine, pm_numbers, ls_numbers, mnist), but {name} was given."
                raise ValueError(name_error)

        # Get the dataset according to dataset_options.
        if dataset_options is not None:
            dataset = get_dataset(**dataset_options)
        else:
            dataset = get_dataset()

        return dataset

    @staticmethod
    def get_iris() -> tuple[np.ndarray, list[str]]:
        """Get the iris dataset.

        :return tuple[np.ndarray, list[str]]: iris dataset
        """
        iris = datasets.load_iris()
        data = iris.data
        labels = list(iris.target)
        for index in range(len(labels)):
            labels[index] = iris.target_names[labels[index]]

        return data, labels

    @staticmethod
    def get_wine() -> tuple[np.ndarray, list[str]]:
        """Get the wine dataset.

        :return tuple[np.ndarray, list[str]]: wine dataset
        """
        wine_data = datasets.load_wine(return_X_y=True)
        data, labels = wine_data
        labels = [f"{label}" for label in labels]
        return data, labels

    @staticmethod
    def get_pm_numbers(
        dimension: int = 4, num_data: int = 100, highest: int = 1
    ) -> tuple[np.ndarray, list[str]]:
        """Get a positive-negative numbers dataset, which is randomly generated.
        The label of each data is "positive" if the summation of the data is non-negative.
        Otherwise, it is "negative".

        :param int dimension: dimension of data, defaults to 4
        :param int num_data: number of data, defaults to 100
        :param int highest: highest number, defaults to 1
        :return tuple[np.ndarray, list[str]]: positive-negative numbers dataset
        """
        data = np.random.rand(num_data, dimension) * highest
        data -= highest / 2
        labels = ["positive" if np.sum(_d) >= 0 else "negative" for _d in data]

        assert len(data) == len(labels)

        return data, labels

    @staticmethod
    def get_ls_numbers(
        dimension: int = 4, num_data: int = 100, highest: int = 1
    ) -> tuple[np.ndarray, list[str]]:
        """Get a large-small numbers dataset, which is randomly generated.
        The label of each data is "large" if all the entries of the data are larger than or equal to
        the half of the given highest. Otherwise, it is "small".
        Note that, in the current implementation, if it is small, then all the entries are smaller
        than the half of the given highest.

        :param int dimension: dimension of data, defaults to 4
        :param int num_data: number of data, defaults to 100
        :param int highest: highest number, defaults to 1
        :return tuple[np.ndarray, list[str]]: large-small numbers dataset
        """
        small_data = np.random.rand(num_data // 2, dimension) * (highest / 2)
        large_data = np.random.rand(num_data // 2, dimension) * (highest / 2)
        large_data += highest / 2
        data = np.concatenate((small_data, large_data))

        small_labels = ["small"] * len(small_data)
        large_labels = ["large"] * len(large_data)
        labels = small_labels + large_labels

        assert len(data) == len(labels)

        return data, labels

    @staticmethod
    def get_hor_ver_images(
        num_images: int = 100,
        image_shape: tuple[int, int] = (2, 4),
        line_length: int = 2,
        line_pixel_value: float = np.pi / 2,
        min_noise_value: float = 0,
        max_noise_value: float = np.pi / 4,
    ) -> tuple[np.ndarray, list[str]]:
        """Generate the line dataset.

        :param tuple[int, int] image_shape: image shape, defaults to (2, 4)
        :param int line_length: length of line, defaults to 2
        :param float line_pixel_value: value of line, defaults to np.pi/2
        :param float min_noise_value: minimum value for noise, defaults to 0
        :param float max_noise_value: maximum value for noise, defaults to np.pi/4
        :param int num_images: number of images
        :return tuple[[np.ndarray, list[str]]: images and their labels
        """

        def __get_all_horizontal_patterns(
            image_shape: tuple[int, int] = (2, 4),
        ) -> np.ndarray:
            """Get all horizontal patterns of the given image shape and length of the line as a flattened array.

            :param tuple[int, int] image_shape: image shape, defaults to (2, 4)
            :return np.ndarray: all horizontal patterns as flattened
            """
            # Make the trivial pattern, which the line is set from the head.
            trivial_pattern = np.zeros(image_shape[1])
            trivial_pattern[:line_length] = line_pixel_value
            # Make the patterns for one line.
            num_patterns_for_one_line = image_shape[1] - (line_length - 1)
            patterns_for_one_line = np.zeros(
                (num_patterns_for_one_line, image_shape[1])
            )
            for index in range(num_patterns_for_one_line):
                patterns_for_one_line[index, :] = np.roll(trivial_pattern, index)

            # Put the patterns for one line to every line.
            num_patterns = num_patterns_for_one_line * image_shape[0]
            image_length = image_shape[0] * image_shape[1]
            patterns = np.zeros((num_patterns, image_length))
            for index in range(image_shape[0]):
                start_row_index = index * num_patterns_for_one_line
                end_row_index = start_row_index + num_patterns_for_one_line
                start_column_index = index * image_shape[1]
                end_column_index = start_column_index + image_shape[1]
                patterns[
                    start_row_index:end_row_index, start_column_index:end_column_index
                ] = patterns_for_one_line

            return patterns

        def __get_all_vertical_patterns(
            image_shape: tuple[int, int] = (2, 4),
        ) -> np.ndarray:
            """Get all vertical patterns of the given image shape and length of the line as a flattened array by using get_all_horizontal_patterns.

            :param tuple[int, int] image_shape: image shape, defaults to (2, 4)
            :return np.ndarray: all vertical patterns as flattened
            """
            # Get all horizontal patterns of the transposed image shape.
            new_image_shape = (image_shape[1], image_shape[0])
            transposed_patterns = __get_all_horizontal_patterns(
                image_shape=new_image_shape
            )
            # Transpose each horizontal pattern so that it is the original vertical pattern.
            patterns = np.zeros(transposed_patterns.shape)
            for index, transposed_pattern in enumerate(transposed_patterns):
                reshaped_transposed_pattern = transposed_pattern.reshape(
                    new_image_shape
                )
                reshaped_pattern = reshaped_transposed_pattern.T
                patterns[index, :] = reshaped_pattern.flatten()

            return patterns

        # Get all horizontal patterns.
        hor_array = __get_all_horizontal_patterns(image_shape=image_shape)

        # Create all vertical line patterns.
        ver_array = __get_all_vertical_patterns(image_shape=image_shape)

        # Generate random images.
        images = []
        labels = []
        for _ in range(num_images):
            rng = np.random.randint(0, 2)
            if rng == 0:
                labels.append("horizontal")
                random_image = np.random.randint(0, hor_array.shape[0])
                images.append(np.array(hor_array[random_image]))
            elif rng == 1:
                labels.append("vertical")
                random_image = np.random.randint(0, ver_array.shape[0])
                images.append(np.array(ver_array[random_image]))

            # Create noise.
            image_length = image_shape[0] * image_shape[1]
            for i in range(image_length):
                if images[-1][i] == 0:
                    images[-1][i] = np.random.uniform(
                        low=min_noise_value, high=max_noise_value
                    )

        return np.array(images), labels

    @staticmethod
    def get_mnist() -> tuple[np.ndarray, list[str]]:
        """Get the MNIST dataset.

        :return tuple[np.ndarray, list[int]]: MNIST dataset
        """
        # Get the MNIST datasets.
        train_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        val_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        val_images = val_dataset.data.numpy()
        val_labels = val_dataset.targets.numpy()

        images = np.expand_dims(np.vstack([train_images, val_images]), axis=1)
        labels = np.concatenate([train_labels, val_labels]).tolist()
        labels = [str(label) for label in labels]
        return images, labels
