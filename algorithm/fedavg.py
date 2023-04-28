from .fedbase import BasicServer, BasicClient
import utils.fflow as flw
import utils.fmodule
import utils.system_simulator as ss
import wandb
import torch.multiprocessing as mp

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def run(self, suffix_log_filename=None):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        if flw.logger.check_exist(suffix_log_filename=suffix_log_filename):
            return
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval):
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        log_filepath = flw.logger.save_output_as_json(suffix_log_filename=suffix_log_filename)
        wandb.save(log_filepath)
        print("LOG FILEPATH", log_filepath)
        return

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        return dict()
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])
        else:
            return None
    
    @ss.with_dropout
    @ss.with_clock
    def communicate(self, selected_clients, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        client_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        for cid in communicate_clients:client_package_buffer[cid] = None
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_threads)
            # for client_id in communicate_clients:
                # self.clients[client_id].update_device(next(utils.fmodule.dev_manager))
                # packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=(int(client_id),)))
            packages_received_from_clients = pool.map(self.communicate_with, communicate_clients)
            import pdb; pdb.set_trace()
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): client_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [client_package_buffer[cid] for cid in selected_clients if client_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
