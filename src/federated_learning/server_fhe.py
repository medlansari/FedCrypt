import copy

from src.data.data_splitter import data_splitter
from src.federated_learning.client import Client


class Server_FHE():

    def __init__(self, model, dataset, nb_clients, id):

        self.dataset = dataset
        self.nb_clients = nb_clients
        self.poly_client = False
        self.poly_server = True
        self.model = model
        self.model.to(DEVICE)
        self.train_subsets, self.subset_size, self.test_set = data_splitter(self.dataset,
                                                                            self.nb_clients
                                                                            )
        self.detector = Detector()
        self.detector.to(DEVICE)
        self.model_test = model
        self.model_test.to(DEVICE)
        self.trigger_set = torch.utils.data.DataLoader(
            WafflePattern(RGB=True, features=True),
            batch_size=10,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.id = id
        self.max_round = 30

        print("Dataset :", dataset)
        print("Number of clients :", self.nb_clients)
        print("Distribution :", self.distrib)

    def train(self, nb_rounds, lr_client, lr_pretrain, lr_retrain):
        print("#" * 60 + "\tFHE\t" + "#" * 60)

        acc_test_list = []
        acc_watermark_black_list = []

        self.pretrain(lr_pretrain)

        print("Number of rounds :", nb_rounds)

        clients = []

        for c in range(self.nb_clients):
            client = Client(self.model.state_dict(), self.train_subsets[c], self.poly_client)

            clients.append(client)

        for r in range(nb_rounds):

            print("")

            selected_clients = random.sample(
                range(self.nb_clients), int(PRCT_TO_SELECT * self.nb_clients)
            )

            loop = tqdm(selected_clients)

            for idx, c in enumerate(loop):
                clients[c].model.load_state_dict(self.model.state_dict())

                clients[c].train(lr=lr_client)

                loop.set_description(f"Round [{r}/{nb_rounds}]")

            self.model = fedavg(np.array(clients), self.subset_size, self.model, self.test_set, selected_clients)

            self.model_test.load_state_dict(self.model.state_dict())

            time_before = time()

            acc_watermark_black = self.retrain(lr_retrain, max_round)

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = test(self.model_test, self.test_set)

            acc_test_list.append(acc_test)

            acc_watermark_black_list.append(acc_watermark_black)

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            lr_retrain = lr_retrain * 0.99

        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(self.secret_key, "./outputs/secret_key_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )

        torch.save(self.message, "./outputs/message_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )

    def train(self, nb_rounds, lr_client, lr_pretrain, lr_retrain, params):
        print("#" * 60 + "\tFHE\t" + "#" * 60)

        lr = lr_retrain

        acc_test_list = []
        acc_watermark_black_list = []

        print("Number of rounds :", nb_rounds)

        watermark_set, dynamic_key = params
        wdr_dynamic = []

        clients = []

        for c in range(self.nb_clients):
            client = Client(self.model.state_dict(), self.train_subsets[c], self.poly_client)

            clients.append(client)

        for r in range(nb_rounds):

            print("")

            selected_clients = random.sample(
                range(self.nb_clients), int(PRCT_TO_SELECT * self.nb_clients)
            )

            loop = tqdm(selected_clients)

            for idx, c in enumerate(loop):
                clients[c].model.load_state_dict(self.model.state_dict())

                clients[c].train(lr=lr_client)

                loop.set_description(f"Round [{r}/{nb_rounds}]")

            self.model = fedavg(np.array(clients), self.subset_size, self.model, self.test_set, selected_clients)

            self.model_test.load_state_dict(self.model.state_dict())

            time_before = time()

            acc_watermark_black = self.retrain(lr_retrain, max_round)

            time_after = time() - time_before

            print("Time for watermark embedding :", round(time_after, 2))

            acc_test, loss_test = test(self.model_test, self.test_set)

            acc_test_list.append(acc_test)

            acc_watermark_black_list.append(acc_watermark_black)

            plot_FHE(
                acc_test_list, acc_watermark_black_list, acc_watermark_white_list, lr_client, lr_pretrain, lr,
                self.id
            )

            print("Accuracy on the test set :", acc_test)
            print("Loss on the test set :", loss_test)

            lr_retrain = lr_retrain * 0.99

            dynamic_key_current = copy.deepcopy(self.detector)

            self.detector = dynamic_key

            _, tmp_dynamic, tmp_static = get_accuracies(self.model, self.test_set, watermark_set, key, message,
                                                        dynamic_key)

            self.detector = dynamic_key_current

            wdr_dynamic.append(tmp_dynamic)
            wdr_static.append(tmp_static)

            plot_overwriting(acc_test_list, acc_watermark_black_list, acc_watermark_white_list,
                             wdr_dynamic, wdr_static)

        torch.save(
            self.model.state_dict(),
            "./outputs/save_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(
            self.detector.state_dict(),
            "./outputs/detector_"
            + str(nb_rounds)
            + "_"
            + str(MAX_EPOCH_CLIENT)
            + "_FHE"
            + "_" + self.id
            + ".pth",
        )

        torch.save(self.secret_key, "./outputs/secret_key_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )

        torch.save(self.message, "./outputs/message_" + str(nb_rounds)
                   + "_"
                   + str(MAX_EPOCH_CLIENT)
                   + "_FHE"
                   + "_" + self.id
                   + ".pth", )


    def pretrain(self, lr_pretrain):
        self.bn_layers_requires_grad(False)

        self.requiere_grad_except_fc3(False)

        optimizer = optim.SGD(self.model.fc1.parameters(), lr=lr_pretrain)  #, weight_decay=1e-4)

        optimizer_detector = optim.SGD(self.detector.parameters(), lr=1e-1)

        criterion = nn.MSELoss()

        acc_watermark_black, _ = test_MSE(self.model, self.detector, self.triggerset)

        acc_watermark_white = self.wb_verify()

        print("Black-Box WDR:", acc_watermark_black)
        print("White-box WDR:", acc_watermark_white)

        print("\n" + "#" * 20 + "\tPretrain Waffle\t" + "#" * 20 + "\n")

        epoch = 0

        w0 = self.model.fc1.weight.data.detach().clone()

        while acc_watermark_black < 1.0:

            accumulate_loss = 0

            for inputs, outputs in self.triggerset:
                optimizer.zero_grad(set_to_none=True)
                optimizer_detector.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                outputs = one_hot_encoding(outputs)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs, True)

                    outputs_predicted = self.detector(outputs_predicted)

                    diff = (1 / 2) * (w0 - self.model.fc1.weight).pow(2).sum()

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    loss = blackbox_loss + (1e-2 * diff)

                loss.backward()

                optimizer.step()
                optimizer_detector.step()

                accumulate_loss += loss.item()

            acc_watermark_black, _ = test_MSE(self.model, self.detector, self.triggerset)
            acc_watermark_white = self.wb_verify()

            print(acc_watermark_black, accumulate_loss)

            epoch += 1

        print("Black-Box WDR:", acc_watermark_black)
        print("White-box WDR:", acc_watermark_white)

        print("\n" + 60 * "#" + "\n")

        self.requiere_grad_except_fc3(True)

        self.bn_layers_requires_grad(True)

        return acc_watermark_black, acc_watermark_white

    def retrain(self, lr_retrain, max_round):
        self.bn_layers_requires_grad(False)

        self.requiere_grad_except_fc3(False)

        print("\n" + "#" * 20 + "\tRetrain Waffle\t" + "#" * 20 + "\n")

        optimizer = optim.SGD(self.model.fc1.parameters(), lr=lr_retrain)

        optimizer_detector = optim.SGD(self.detector.parameters(), lr=1e-2)

        criterion = nn.MSELoss()

        acc_watermark_black_before, loss_bb = test_MSE(self.model, self.detector, self.triggerset)

        acc_watermark_white_before = self.wb_verify()

        print("Black-Box WDR:", acc_watermark_black_before, loss_bb)
        print("White-box WDR:", acc_watermark_white_before)

        loop = tqdm(list(range(max_round)))

        for idx, epoch in enumerate(loop):

            accumulate_loss = 0

            for inputs, outputs in self.triggerset:
                optimizer.zero_grad(set_to_none=True)
                optimizer_detector.zero_grad(set_to_none=True)

                inputs = inputs.to(DEVICE, memory_format=torch.channels_last)

                outputs = outputs.to(DEVICE)

                outputs = one_hot_encoding(outputs)

                with torch.autocast(device_type="cuda"):
                    outputs_predicted = self.model(inputs, True)

                    outputs_predicted = self.detector(outputs_predicted)

                    blackbox_loss = criterion(outputs_predicted, outputs)

                    regul = 1e-2 * torch.mean(self.model.fc1.weight, dim=0).pow(2).sum()

                    loss = blackbox_loss + regul

                loss.backward()

                optimizer.step()
                optimizer_detector.step()

                accumulate_loss += loss.item()

            acc_watermark_black, loss_watermark = test_MSE(self.model, self.detector, self.triggerset)

            acc_watermark_white = self.wb_verify()

            loop.set_description(f"Epoch [{epoch}/{max_round}]")

            loop.set_postfix(
                {
                    "Black-Box WDR :": acc_watermark_black,
                    "White-Box WDR :": acc_watermark_white,
                    "Watermarking Loss ": loss_watermark,
                }
            )

        print("\n" + 60 * "#" + "\n")

        self.requiere_grad_except_fc3(True)

        self.bn_layers_requires_grad(True)

        return acc_watermark_black_before, acc_watermark_white_before
