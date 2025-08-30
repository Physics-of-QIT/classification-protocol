from qml3_library import *
from qml3_viz_library import *


def executor(
        args,
        train=False,
        test1=False,
        test2=False,
):
    # -------------------------------------
    # Parse Parameters
    # -------------------------------------
    dataset = args.dataset  # Dataset
    idx_run = args.idx_run  # Index of runs
    num_classes = args.num_classes  # Number of classes
    num_qubits = args.num_qubits  # Number of qubits
    time = args.time  # Time step
    learning_rate = args.learning_rate  # Learning rate
    weight_decay = args.weight_decay  # Weight decay
    epochs = args.epochs  # Number of epochs
    batch_size = args.batch_size  # Batch size
    num_train_samples = args.num_train_samples  # Number of train samples
    num_test1_samples = args.num_test1_samples  # Number of test1 samples
    num_test2_samples = args.num_test2_samples  # Number of test2 samples
    exp = args.exp  # Experiment

    gamma = 1.0

    init = torch.zeros((2 ** num_qubits, 1), dtype=torch.complex64, device=device)
    init[0, 0] = 1

    condition_train, condition_test1, condition_test2 = get_conditions(args)
    train = True if (
            train and not (data_dir / args.dataset  / f"train_accuracy{condition_train}.pt").exists()
    ) else False
    test1 = True if (
            test1 and not (data_dir / args.dataset  / f"test1_accuracy{condition_test1}.pt").exists()
    ) else False
    test2 = True if (
            test2 and not (data_dir / args.dataset  / f"test2_accuracy{condition_test2}.pt").exists()
    ) else False
    logging.debug(f"Train: {train}, Test1: {test1}, Test2: {test2}")

    # -------------------------------------
    # Load Training & Testing Data
    # -------------------------------------
    if dataset == "SAT4" or dataset == "SAT6":
        # Data Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Create the dataset
        train_dataset = SAT(
            num_classes=num_classes,
            split="train",
            root=dataset_dir,
            transform=transform,
        )
        test_dataset = SAT(
            num_classes=num_classes,
            split="test",
            root=dataset_dir,
            transform=transform,
        )

        # Create the paired train dataset
        train_dataset = PairedDataset(train_dataset, num_train_samples)

        # Create the paired test1 dataset
        test1_dataset = filter_data_by_class(test_dataset, range(num_classes), num_test1_samples // num_classes)
        test1_dataset = PairedDataset(test1_dataset, num_test1_samples)

        # Create the test2 dataset
        test2_dataset = filter_by_number(test_dataset, num_test2_samples)

    elif dataset == "BloodMNIST":
        # Data Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Create the dataset
        train_dataset = BloodMNIST(
            split="train",
            transform=transform,
            download=False,
            as_rgb=False,
            size=28,
            root=dataset_dir / 'medmnist',
        )
        test1_dataset = BloodMNIST(
            split="val",
            transform=transform,
            download=False,
            as_rgb=False,
            size=28,
            root=dataset_dir / 'medmnist',
        )
        test2_dataset = BloodMNIST(
            split="test",
            transform=transform,
            download=False,
            as_rgb=False,
            size=28,
            root=dataset_dir / 'medmnist',
        )

        # Create the paired train dataset
        train_dataset = PairedDataset(train_dataset, num_train_samples)

        # Create the paired test1 dataset
        test1_dataset = filter_data_by_class(test1_dataset, range(num_classes), num_test1_samples // num_classes)
        test1_dataset = PairedDataset(test1_dataset, num_test1_samples)

        # Create the test2 dataset
        test2_dataset = filter_by_number(test2_dataset, num_test2_samples)

    elif dataset == "MNIST":
        # Data Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Load the MNIST dataset
        train_dataset = datasets.MNIST(
            root=dataset_dir, train=True, transform=transform, download=False,
        )
        test_dataset = datasets.MNIST(
            root=dataset_dir, train=False, transform=transform, download=False,
        )

        # Create the paired train dataset
        # Filter the dataset for specified classes
        train_dataset = filter_by_label(train_dataset, range(num_classes))
        train_dataset = PairedDataset(train_dataset, num_train_samples)

        # Create the paired test1 dataset
        # Filter the dataset for specified classes
        test1_dataset = filter_by_label(test_dataset, range(num_classes))
        # Filter the dataset for specified number of samples
        test1_dataset = filter_data_by_class(test1_dataset, range(num_classes), num_test1_samples // num_classes)
        test1_dataset = PairedDataset(test1_dataset, num_test1_samples)

        # Create the test2 dataset
        # Filter the dataset for specified classes
        test2_dataset = filter_by_label(test_dataset, range(num_classes))
        # Filter the dataset for specified number of samples
        test2_dataset = filter_by_number(test2_dataset, num_test2_samples)

    elif dataset == "FashionMNIST":
        # Data Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Load the FashionMNIST dataset
        train_dataset = datasets.FashionMNIST(
            root=dataset_dir, train=True, transform=transform, download=False,
        )
        test_dataset = datasets.FashionMNIST(
            root=dataset_dir, train=False, transform=transform, download=False,
        )

        # Create the paired train dataset
        train_dataset = filter_by_label(train_dataset, range(num_classes))
        train_dataset = PairedDataset(train_dataset, num_train_samples)

        # Create the paired test1 dataset
        test1_dataset = filter_by_label(test_dataset, range(num_classes))
        test1_dataset = filter_data_by_class(test1_dataset, range(num_classes), num_test1_samples // num_classes)
        test1_dataset = PairedDataset(test1_dataset, num_test1_samples)

        # Create the test2 dataset
        test2_dataset = filter_by_label(test_dataset, range(num_classes))
        test2_dataset = filter_by_number(test2_dataset, num_test2_samples)

    elif dataset == "GTSRBD":
        # Data Transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize all images to 32*32
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Create the dataset
        dataset = GTSRBDataset(
            root_dir=dataset_dir / 'GTSRBD' / 'Final_Training',
            num_classes=num_classes,
            transform=transform,
        )

        # Create the paired train dataset
        train_dataset = PairedDataset(dataset, num_train_samples)

        # Create the paired test1 dataset
        test1_dataset = filter_data_by_class(dataset, range(num_classes), num_test1_samples // num_classes)
        test1_dataset = PairedDataset(test1_dataset, num_test1_samples)

        # Create the test2 dataset
        test2_dataset = filter_by_number(dataset, num_test2_samples)

    else:
        raise ValueError(f"Unknown dataset: {dataset}.")

    # Log dataset sizes
    logging.debug(f"Paired Train Dataset: {len(train_dataset)}")
    logging.debug(f"Paired Test1 Dataset: {len(test1_dataset)}")
    logging.debug(f"Test2 Dataset: {len(test2_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test1_loader = DataLoader(test1_dataset, batch_size=1, shuffle=True)
    test2_loader = DataLoader(test2_dataset, batch_size=1, shuffle=True)

    # Display some samples
    if idx_run == -1:
        viz_samples(
            args, test2_loader, samples_per_class=5,
            save=True,
        )
        return -1, -1, -1

    # -------------------------------------
    # Define the Model
    # -------------------------------------
    model = ComparisonNet(
        args,
        gamma,
        init,
    ).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # -------------------------------------
    # Training Phase
    # -------------------------------------
    if train:
        logging.info("Training the model.")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            for img_pairs, _, labels in train_loader:
                img_pairs[0], img_pairs[1], labels = img_pairs[0].to(device), img_pairs[1].to(device), labels.to(device)

                # Forward pass
                outputs = model(img_pairs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (abs(outputs - labels) < 0.5).int().sum().item()  # Count correct predictions
            train_accuracy = (correct / len(train_loader) / batch_size) * 100  # Convert to percentage

            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%"
            )

        # save the results
        torch.save(
            train_accuracy,
            data_dir / args.dataset / f"train_accuracy{condition_train}.pt",
        )
        torch.save(
            model.state_dict(),
            data_dir / args.dataset / f"model{condition_train}.pt",
        )
    else:
        train_accuracy = torch.load(data_dir / args.dataset / f"train_accuracy{condition_train}.pt")
        logging.info(f"Skipping training. Train Accuracy: {train_accuracy:.2f}%")
        if test1 or test2:
            model.load_state_dict(
                torch.load(data_dir / args.dataset / f"model{condition_train}.pt", map_location=device))

    # -------------------------------------
    # Evaluation Phase
    # -------------------------------------
    if test1:
        logging.info("Testing the comparator.")
        model.eval()

        # Initialize the storage
        hFields_dict = {key: [] for key in range(num_classes)}
        hFields_avg_dict = {key: [] for key in range(num_classes)}
        fin_dict = {key: [] for key in range(num_classes)}
        observables_dict = {key: [] for key in range(num_classes)}

        correct = 0  # Counter for correct predictions
        total_loss = []  # List to track loss during evaluation
        with torch.no_grad():
            for idx, (img_pairs, label_pairs, labels) in enumerate(test1_loader):
                img_pairs[0], img_pairs[1], labels = img_pairs[0].to(device), img_pairs[1].to(device), labels.to(device)

                outputs = model(img_pairs)
                loss = criterion(outputs, labels)

                logging.debug(f"NO.{idx} Test1 Predictions: {outputs} | Labels: {labels}")
                for idy, label in enumerate(label_pairs[0]):
                    fin_dict[label.item()].append(model.fin1_list[-1 - idy])
                    hFields_dict[label.item()].append(model.hFields1_list[-1 - idy])
                for idy, label in enumerate(label_pairs[1]):
                    fin_dict[label.item()].append(model.fin2_list[-1 - idy])
                    hFields_dict[label.item()].append(model.hFields2_list[-1 - idy])

                correct += (abs(outputs - labels) < 0.5).int().sum().item()
                total_loss.append(loss.item())

        test1_loss = sum(total_loss) / len(total_loss)
        logging.info("Test1 Loss: {:.4f}".format(test1_loss))
        test1_accuracy = (correct / len(test1_loader) / 1) * 100  # Convert to percentage
        logging.info(f"Test1 Accuracy: {test1_accuracy:.2f}%")

        # -------------------------------------
        # Store the Results
        # -------------------------------------
        for fin1, fin2 in zip(model.fin1_list, model.fin2_list):
            logging.debug(torch.abs(fin2.conj().T @ fin1) ** 2)
        for idx_batch, (_, labels_pair, labels) in enumerate(test1_loader):
            logging.debug(f"Batch Index: {idx_batch}, Labels: {labels}, Labels Pair: {labels_pair}")

        for label, tensor in hFields_dict.items():
            stacked_tensors = torch.stack(tensor)
            hFields_dict[label] = stacked_tensors
            hFields_avg_dict[label] = stacked_tensors.mean(dim=0)
        for label, tensor in fin_dict.items():
            stacked_tensors = torch.stack(tensor)
            fin_dict[label] = stacked_tensors.mean(dim=0)

            observables = [fin @ fin.conj().T for fin in tensor]
            stacked_observables = torch.stack(observables)
            observables_dict[label] = stacked_observables.mean(dim=0)
        observables_dict = dict(sorted(observables_dict.items()))

        # save the results
        torch.save(
            test1_accuracy,
            data_dir / args.dataset / f"test1_accuracy{condition_test1}.pt",
        )
        torch.save(
            hFields_dict,
            data_dir / args.dataset / f"hFields_dict{condition_test1}.pt",
        )
        torch.save(
            hFields_avg_dict,
            data_dir / args.dataset / f"hFields_avg_dict{condition_test1}.pt",
        )
        torch.save(
            fin_dict,
            data_dir / args.dataset / f"fin_dict{condition_test1}.pt",
        )
        torch.save(
            observables_dict,
            data_dir / args.dataset / f"observables_dict{condition_test1}.pt",
        )
    else:
        test1_accuracy = torch.load(data_dir / args.dataset / f"test1_accuracy{condition_test1}.pt")
        logging.info(f"Skipping testing of the classifier. Test1 Accuracy: {test1_accuracy:.2f}%")
        if test2:
            observables_dict = torch.load(data_dir / args.dataset / f"observables_dict{condition_test1}.pt")

    if test2:
        logging.info("Testing the classifier.")
        model.eval()

        # Initialize the storage
        overlaps_dict = {key: [] for key in range(num_classes)}
        overlaps_avg_dict = {key: [] for key in range(num_classes)}

        # Set observables from database
        model.observables = observables_dict.values()

        correct = 0  # Counter for correct predictions
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(test2_loader):
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model.evaluate(imgs)

                logging.debug(f"NO.{idx} Test2 Predictions: {outputs} | Labels: {labels}")
                for idy, label in enumerate(labels):
                    overlaps_dict[label.item()].append(model.overlaps_list[-1 - idy])

                correct += (outputs == labels).int().sum().item()

        test2_accuracy = (correct / len(test2_loader) / 1) * 100  # Convert to percentage
        logging.info(f"Test2 Accuracy: {test2_accuracy:.2f}%")

        # -------------------------------------
        # Store the Results
        # -------------------------------------
        for label, tensor in overlaps_dict.items():
            stacked_tensors = torch.stack(tensor)
            overlaps_dict[label] = stacked_tensors
            overlaps_avg_dict[label] = stacked_tensors.mean(dim=0)

        # save the results
        torch.save(
            test2_accuracy,
            data_dir / args.dataset / f"test2_accuracy{condition_test2}.pt",
        )
        torch.save(
            overlaps_dict,
            data_dir / args.dataset / f"overlaps_dict{condition_test2}.pt",
        )
        torch.save(
            overlaps_avg_dict,
            data_dir / args.dataset / f"overlaps_avg_dict{condition_test2}.pt",
        )
    else:
        test2_accuracy = torch.load(data_dir / args.dataset / f"test2_accuracy{condition_test2}.pt")
        logging.info(f"Skipping testing of the classifier. Test2 Accuracy: {test2_accuracy:.2f}%")

    return train_accuracy, test1_accuracy, test2_accuracy
