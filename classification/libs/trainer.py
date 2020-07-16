import os
import re
import sys
import glob
import json
import copy
import torch
import inspect
import functools
from termcolor import colored

from tqdm import tqdm
from time import gmtime, strftime
from libs.utils import confusion_matrix_fig
from libs.arg_types import arg_boolean, arg_dict
from libs.tensorboardXsafe import SummaryWriter

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import numpy as np
from sklearn import metrics


def add_train_params(parser):
    parser.add('-c', '--config', is_config_file=True)
    parser.add_argument('--sacred', type=arg_boolean, default=False)
    parser.add_argument('--mongodb_disable', type=arg_boolean, default=False)
    parser.add_argument('--mongodb_url', type=str, default='127.0.0.1')
    parser.add_argument('--mongodb_port', type=str, default='27017')
    parser.add_argument('--mongodb_name', type=str, default=None)

    parser.add_argument('--iterate_seed', nargs='+', type=int, default=[])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--val_perc', type=float)

    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', nargs='+', type=int, default=32)
    parser.add_argument('--optimize_every', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_param_multipliers', type=arg_dict(float))
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--max_beaten_epochs', type=int, default=4)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--keep_only_best_checkpoint', type=arg_boolean, default=False)

    parser.add_argument('--use_chunks', nargs='+', type=arg_boolean, default=False)
    parser.add_argument('--chunks_delta_t', type=int, default=5e4)
    parser.add_argument('--chunks_min_delta_t', type=int, default=2.5e4)
    parser.add_argument('--chunks_min_n_events', type=int, default=150)

    return parser


class DisableGradNotScriptContext:
    def __init__(self, model):
        self.model = model
        self.script = self.has_scriptmodule(self.all_submodules(model))
        self.context = None

    @staticmethod
    def all_submodules(module):
        submod = list(module.modules())
        for m in module.modules():
            if m != module:
                submod += DisableGradNotScriptContext.all_submodules(m)
        return submod

    @staticmethod
    def has_scriptmodule(module_list):
        for mod in module_list:
            if isinstance(mod, torch.jit.ScriptModule):
                return True
        return False

    def __enter__(self):
        if not self.script:
            self.context = torch.no_grad()
            self.context.__enter__()

    def __exit__(self, *args):
        if not self.script:
            self.context.__exit__(*args)

# Rename class for convenience
no_grad_ifnotscript = DisableGradNotScriptContext


def main_ifsacred(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Check at run-time if sacred has to be used
        if self.use_sacred:
            # If this is the case, wrap the function
            @self.sacred_exp.main
            def decor_func():
                return func(*args, **kwargs)

            # and run it through sacred run()
            run = self.sacred_exp.run()
            return run.result
        else:
            # Otherwise just call the function
            return func(*args, **kwargs)

    return wrapper


def seed_iterator(params):

    num_eval = len(params.iterate_seed)
    if num_eval > 1:
        print(colored("Running multiple seed evaluation. Num iterations: {}".format(num_eval), "green"))
    else:
        yield params

    for i, seed in enumerate(params.iterate_seed):
        params_copy = copy.deepcopy(params)
        params_copy.seed = seed
        if num_eval > 1:
            params_copy.exp_name += "_seed%d" % seed

        yield params_copy


class Trainer:
    def __init__(self, network, optim_class, train_loader,
                 val_loader, test_loader, params,
                 lr_scheduler=None, params_scheduler=None,
                 verbose=True):

        super().__init__()
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = train_loader.dataset.num_classes
        self.class_names = train_loader.dataset.classes
        self.num_weights = self.get_num_weights()
        self.num_trainable_weights = self.get_num_weights(trainable=True)

        self.verbose = verbose
        self.params = params
        self.seed = params.seed
        self.restart_path = params.restart_path
        self.max_epochs = params.max_epochs
        self.max_beaten_epochs = params.max_beaten_epochs
        self.learning_rate = params.learning_rate
        self.lr_param_multipliers = params.lr_param_multipliers
        self.optimize_every = params.optimize_every
        self.clip_grad_norm = params.clip_grad_norm
        self.keep_only_best_checkpoint = params.keep_only_best_checkpoint

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % params.exp_name
        self.log_dir = self.restart_path if self.restart_path else os.path.join(params.log_dir, timestring)

        self.use_sacred = params.sacred
        if params.sacred:
            self.sacred_exp = Experiment(params.exp_name)
            self.sacred_exp.captured_out_filter = apply_backspaces_and_linefeeds
            configs = vars(params)
            configs.update({'num_weights': self.num_weights})
            configs.update({'num_trainable_weights': self.num_trainable_weights})
            configs.update({'log_dir': self.log_dir})
            self.sacred_exp.add_config(self.mongo_compatible(configs))
            for source in self.get_sources():
                self.sacred_exp.add_source_file(source)

            if not params.mongodb_disable:
                url = "{0.mongodb_url}:{0.mongodb_port}".format(params)
                db_name = [d for d in params.data_dir.split('/') if len(d) > 0][-1]
                if hasattr(params, 'mongodb_name') and params.mongodb_name:
                    db_name = params.mongodb_name

                print(colored('Connect to MongoDB@{}:{}'.format(url, db_name), "green"))
                self.sacred_exp.observers.append(MongoObserver.create(url=url, db_name=db_name))

        self.seen = 0
        self.epoch = 0
        self.steps = 0
        self.best_epoch = 0
        self.best_epoch_score = 0
        self.beaten_epochs = 0
        self.optim_class = optim_class
        self.lr_scheduler = lr_scheduler
        self.params_scheduler = params_scheduler
        self.optimizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @classmethod
    def mongo_compatible(cls, obj):
        if isinstance(obj, dict):
            res = dict()
            for key, value in obj.items():
                # '.' and '$' are mongoDB reserved characters, replace them
                # as ',' and '£' respectively
                key = key.replace(".", ',').replace("$", "£")
                res[key] = cls.mongo_compatible(value)
            return res
        elif isinstance(obj, (list, tuple)):
            return list([cls.mongo_compatible(value) for value in obj])
        return obj

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(colored("Using seed %d" % seed, "green"))

    def get_sources(self):
        sources = []
        # The network file
        sources.append(inspect.getfile(self.network.__class__))
        # the main script
        sources.append(sys.argv[0])
        # and any user custom submodule
        for module in self.network.children():
            module_path = inspect.getfile(module.__class__)
            if 'site-packages' not in module_path:
                sources.append(module_path)

        # The configuration file
        # if hasattr(self.params, "config") and self.params.config:
        #     if os.path.exists(self.params.config):
        #         sources.append(self.params.config)

        return sources

    def get_num_weights(self, trainable=False):
        return sum([p.numel() for p in self.network.parameters() if not trainable or p.requires_grad])

    def log_sacred_scalar(self, name, val, step):
        if self.use_sacred and hasattr(self, 'sacred_exp') and self.sacred_exp.current_run:
            self.sacred_exp.current_run.log_scalar(name, val, step)

    def log_params(self, logger):
        name_str = os.path.basename(sys.argv[0])
        args_str = "".join([("  %s: %s \n" % (arg, val)) for arg, val in sorted(vars(self.params).items())])[:-2]
        logger.add_text("Script arguments", name_str + "\n" + args_str)

    def json_params(self, savedir):
        try:
            dict_params = vars(self.params)
            json_path = os.path.join(savedir, "params.json")

            with open(json_path, 'w') as fp:
                json.dump(dict_params, fp)
        except Exception as e:
            print(colored("An error occurred while saving parameters into JSON:", "red"))
            print(e)

    def yaml_params(self, savedir):
        try:
            yaml_path = os.path.join(savedir, "params.yaml")
            with open(yaml_path, 'w') as fp:
                for k, v in vars(self.params).items():
                    if isinstance(v, list):
                        v = "[" + "".join(["{}, ".format(z) for z in v])[:-len(", ")] + "]"
                    if isinstance(v, str) and len(v) == 0:
                        continue
                    if v is None:
                        continue
                    fp.write("{}: {}\n".format(k, v))
        except Exception as e:
            print(colored("An error occurred while saving parameters into YAML:", "red"))
            print(e)

    def json_results(self, savedir, test_score):
        try:
            json_path = os.path.join(savedir, "results.json")
            results = {'seen': self.seen,
                       'epoch': self.epoch,
                       'best_epoch': self.best_epoch,
                       'beaten_epochs': self.beaten_epochs,
                       'best_epoch_score': self.best_epoch_score,
                       'test_score': test_score}

            with open(json_path, 'w') as fp:
                json.dump(results, fp)

            if self.use_sacred and hasattr(self, 'sacred_exp') and self.sacred_exp.current_run:
                self.sacred_exp.current_run.add_artifact(json_path)
        except Exception as e:
            print("An error occurred while saving results into JSON:")
            print(e)

    def log_gradients(self, logger, global_step):

        for name, param in self.network.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.add_scalar("gradients/"+name, param.grad.norm(2).item(), global_step=global_step)

    def restart_exp(self):
        regex = re.compile(r'.*epoch(\d+)\.ckpt')
        checkpoints = glob.glob(os.path.join(self.restart_path, "*.ckpt"))
        # Sort checkpoints
        checkpoints = sorted(checkpoints, key=lambda f: int(regex.findall(f)[0]))
        last_checkpoint = checkpoints[-1]
        self.load_trainer_checkpoint("", last_checkpoint)

    def save_checkpoint(self, path, filename):
        os.makedirs(path, exist_ok=True)

        try:
            torch.save({'seen': self.seen,
                        'epoch': self.epoch,
                        'best_epoch': self.best_epoch,
                        'beaten_epochs': self.beaten_epochs,
                        'best_epoch_score': self.best_epoch_score,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                        }, os.path.join(path, filename))
        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)

    def remove_checkpoint(self, path, filename):
        try:
            # The checkpoint may already be removed
            os.remove(os.path.join(path, filename))
        except:
            pass

    def load_trainer_checkpoint(self, path, filename):
        ckpt_path = os.path.join(path, filename)

        checkpoint = torch.load(ckpt_path)
        self.seen = checkpoint['seen']
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.beaten_epochs = checkpoint['beaten_epochs']
        self.best_epoch_score = checkpoint['best_epoch_score']
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_network_checkpoint(self, path, filename):
        ckpt_path = os.path.join(path, filename)

        checkpoint = torch.load(ckpt_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])

    def correct(self, batch_probas, batch_labels):
        """
        Computes the number of correct predictions given
        :param batch_probas: A tensor of shape [batch_size, num_classes] containing the predicted probabilities
        :param batch_labels: A tensor of shape [num_classes] containing the target labels
        :return: a scalar representing the number of correct predictions
        """

        predicted_classes = torch.argmax(batch_probas, dim=-1)
        return torch.sum(predicted_classes == batch_labels)

    def confusion_matrix(self, batch_probas, batch_labels):
        """

        :param batch_probas:
        :param batch_labels:
        :return:
        """

        _, batch_predictions = torch.max(batch_probas, dim=-1)
        if isinstance(batch_predictions, torch.Tensor):
            batch_predictions = batch_predictions.clone().cpu().data.numpy()
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.clone().cpu().data.numpy()
        conf_matrix = metrics.confusion_matrix(batch_labels, batch_predictions,
                                               labels=np.arange(self.num_classes))
        return conf_matrix

    def optimizer_parameters(self, base_lr, params_mult):
        """
        Associates network parameters with learning rates
        :param float base_lr: the basic learning rate
        :param OrderedDict params_mult: an OrderedDict containing 'param_name':lr_multiplier pairs. All parameters containing
        'param_name' in their name are be grouped together and assigned to a lr_multiplier*base_lr learning rate.
        Parameters not matching any 'param_name' are assigned to the base_lr learning_rate
        :return: A list of dictionaries [{'params': <list_params>, 'lr': lr}, ...]
        """

        selected = []
        grouped_params = []
        if params_mult is not None:
            for groupname, multiplier in params_mult.items():
                group = []
                for paramname, param in self.network.named_parameters():
                    if groupname in paramname:
                        if paramname not in selected:
                            group.append(param)
                            selected.append(paramname)
                        else:
                            raise RuntimeError("%s matches with multiple parameters groups!" % paramname)
                if group:
                    grouped_params.append({'params': group, 'lr': multiplier * base_lr})

        other_params = [param for paramname, param in self.network.named_parameters() if paramname not in selected]
        grouped_params.append({'params': other_params, 'lr': base_lr})
        assert len(selected)+len(other_params) == len(list(self.network.parameters()))

        return grouped_params

    @main_ifsacred
    def train_network(self):
        """
        Performs a complete training procedure by performing early stopping using the provided validation set
        """

        self.seen = 0
        self.steps = 0
        self.epoch = 0
        self.best_epoch = 0
        self.beaten_epochs = 0
        self.best_epoch_score = 0
        # Initializes the network and optimizer states
        self.set_seed(self.seed)
        self.network.init_params()
        # Moves the network to 'device' (GPU if available)
        self.network = self.network.to(self.device)

        self.optimizer = self.optim_class(self.optimizer_parameters(base_lr=self.learning_rate,
                                                                    params_mult=self.lr_param_multipliers),
                                          lr=self.learning_rate)
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.lr_scheduler(self.optimizer)
        # Loads the initial checkpoint if provided
        if self.restart_path is not None:
            self.restart_exp()

        # Use the same directory as the restart experiment
        os.makedirs(self.log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=self.log_dir)
        self.log_params(logger)
        self.json_params(self.log_dir)
        self.yaml_params(self.log_dir)

        while self.beaten_epochs < self.max_beaten_epochs and self.epoch < self.max_epochs:

            if self.lr_scheduler is not None:
                for i, lr in enumerate(self.lr_scheduler.get_lr()):
                    logger.add_scalar("train/lr%d" % i, lr, self.epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.params_scheduler is not None:
                self.params_scheduler.step()

            # Performs one epoch of training
            self.train_epoch(self.train_loader, logger=logger)
            self.save_checkpoint(self.log_dir, "epoch%d.ckpt" % self.epoch)
            # Validate the network against the validation set
            valid_score = self.validate_network(self.val_loader, log_descr="validation", logger=logger)

            if valid_score > self.best_epoch_score or self.epoch == 0:
                if self.keep_only_best_checkpoint and self.epoch != 0:
                    self.remove_checkpoint(self.log_dir, "epoch%d.ckpt" % self.best_epoch)
                    self.remove_checkpoint(self.log_dir, "epoch%d.ckpt" % (self.epoch - 1))
                self.best_epoch = self.epoch
                self.best_epoch_score = valid_score
                self.beaten_epochs = 0
            else:
                if self.keep_only_best_checkpoint and self.beaten_epochs > 0:
                    self.remove_checkpoint(self.log_dir, "epoch%d.ckpt" % (self.epoch - 1))
                self.beaten_epochs += 1

            self.epoch += 1

        self.load_network_checkpoint(self.log_dir, "epoch%d.ckpt" % self.best_epoch)
        test_score = self.validate_network(self.test_loader, log_descr="test", logger=logger)
        self.json_results(self.log_dir, test_score)
        self.log_sacred_scalar("test/accuracy", test_score, self.epoch)

        return test_score

    def train_epoch(self, dataloader, logger=None):
        """
        Performs one entire epoch of training
        :param dataloader: A DataLoader object producing training samples
        :return: a tuple (epoch_loss, epoch_accuracy)
        """

        running_loss = 0
        running_correct = 0
        running_samples = 0
        running_batches = 0
        running_iter_loss = 0
        running_iter_correct = 0
        running_iter_samples = 0

        tot_num_batches = len(dataloader)
        running_optimize_every = min(self.optimize_every, tot_num_batches - running_batches)

        # Enters train mode
        self.network.train()
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        if not self.verbose:
            print("Starting training phase ...")

        pbar_descr_prefix = "Epoch %d (best: %d, beaten: %d)" % (self.epoch, self.best_epoch, self.beaten_epochs)
        with tqdm(total=tot_num_batches,
                  disable=not self.verbose,
                  desc=pbar_descr_prefix + " - Mini-batch progress") as iterator:
            for batch in dataloader:
                # Get the inputs
                batch_lengths, batch_events, batch_labels = batch
                # Moves batch to the proper device based on GPU availability
                batch_lengths = batch_lengths.to(self.device)
                batch_events = batch_events.to(self.device).type(torch.float32)
                batch_labels = batch_labels.to(self.device)

                # forward + backward + optimize
                batch_outputs = self.network.forward(batch_events, batch_lengths)
                loss = self.network.loss(batch_outputs, batch_labels)
                norm_loss = loss / running_optimize_every
                norm_loss.backward()

                loss_b = loss.item()
                running_loss += loss_b
                running_iter_loss += loss_b
                correct_b = self.correct(batch_outputs, batch_labels).item()
                running_correct += correct_b
                running_iter_correct += correct_b

                running_batches += 1
                samples_b = batch_labels.shape[0]
                self.seen += samples_b
                running_samples += samples_b
                running_iter_samples += samples_b

                if running_batches % running_optimize_every == 0:
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

                    self.log_gradients(logger, global_step=self.steps)
                    self.network.log_parameters(logger, global_step=self.steps)
                    logger.add_scalar("train/loss", running_iter_loss / running_optimize_every,
                                      global_step=self.seen)
                    logger.add_scalar("train/accuracy", running_iter_correct / running_iter_samples,
                                      global_step=self.seen)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Either self.optimize_every or the number of remaining samples
                    running_optimize_every = min(self.optimize_every, tot_num_batches - running_batches)
                    running_iter_samples = 0
                    running_iter_correct = 0
                    running_iter_loss = 0
                    self.steps += 1

                # Only print infos at 25%, 50%, 75%
                if not self.verbose and self.steps % (tot_num_batches // 4) == 0:
                    print(("Training phase {}% - " + pbar_descr_prefix + " - metric: {},  loss: {}")
                          .format(int((running_batches / tot_num_batches) * 100),
                                  running_correct / running_samples,
                                  running_loss / running_batches))

                iterator.update()

            tot_metric = running_correct / running_samples
            tot_loss = running_loss / running_batches
            iterator.set_description(pbar_descr_prefix + " - metric: %f,  loss: %f" % (tot_metric, tot_loss))
            self.log_sacred_scalar("train/accuracy", tot_metric, self.epoch)

        if not self.verbose:
            print("Finished " + pbar_descr_prefix + " - metric: %f,  loss: %f" % (tot_metric, tot_loss))

        return tot_loss, tot_metric

    def validate_network(self, dataloader, log_descr="validation", logger=None):
        """
        Computes the accuracy of the network against a validation set
        :param dataloader: A DataLoader object producing validation/test samples
        :return: the accuracy over the validation dataset
        """

        running_samples = 0
        running_correct = 0
        running_confmatrix = np.zeros([self.num_classes, self.num_classes], dtype=np.int)

        if not self.verbose:
            print("Starting "+log_descr+" phase ...")

        # Enters eval mode
        self.network.eval()
        # Disable autograd while evaluating the model
        with no_grad_ifnotscript(self.network):
            with tqdm(total=len(dataloader),
                      disable=not self.verbose,
                      desc="    %s - Mini-batch progress" % log_descr.title()) as iterator:
                for batch in dataloader:
                    # Get the inputs
                    batch_lengths, batch_events, batch_labels = batch
                    # Moves batch to the proper device based on GPU availability
                    batch_lengths = batch_lengths.to(self.device)
                    batch_events = batch_events.to(self.device).type(torch.float32)
                    batch_labels = batch_labels.to(self.device)

                    batch_outputs = self.network.forward(batch_events, batch_lengths)
                    running_correct += self.correct(batch_outputs, batch_labels).item()
                    running_confmatrix += self.confusion_matrix(batch_outputs, batch_labels)
                    running_samples += batch_labels.shape[0]

                    iterator.update()

                tot_metric = running_correct / running_samples
                logger.add_scalar(log_descr + "/accuracy", tot_metric, global_step=self.epoch)
                self.log_sacred_scalar((log_descr + "/accuracy"), tot_metric, self.epoch)
                confmatrix_fig = confusion_matrix_fig(running_confmatrix, self.class_names)
                logger.add_figure(log_descr + "/confusion_matrix", confmatrix_fig, global_step=self.epoch, close=True)
                self.network.log_validation(logger, global_step=self.epoch)

                iterator.set_description("    %s - metric: %f" % (log_descr.title(), tot_metric))

        if not self.verbose:
            print("Finished %s - metric: %f" % (log_descr, tot_metric))

        return tot_metric
