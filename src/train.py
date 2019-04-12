import time
from copy import deepcopy
from random import shuffle

from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from utils import utils_plot as util_plot
from utils.input import *
from utils.utils_train import Tracker


def run_train_dafi(dafi_tree, hparams, input):
    """
    train a classifier on the top of DAFi features
    :param dafi_tree:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    if hparams['optimizer'] == "SGD":
        dafi_optimizer_classifier = torch.optim.SGD([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                    lr=hparams['learning_rate_classifier'])
    else:
        dafi_optimizer_classifier = torch.optim.Adam([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                     lr=hparams['learning_rate_classifier'])

    for epoch in range(hparams['n_epoch_dafi']):
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]
        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]
            dafi_optimizer_classifier.zero_grad()
            output = dafi_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            dafi_optimizer_classifier.step()
    print("Running time for training classifier with DAFi gates: %.3f seconds." % (time.time() - start))
    return dafi_tree


def run_train_model(model_tree, hparams, input):
    """

    :param model_tree:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = [p for p in model_tree.parameters() if p not in classifier_params]
    if hparams['optimizer'] == "SGD":
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.SGD(gates_params, lr=hparams['learning_rate_gates'])
    else:
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.Adam(gates_params, lr=hparams['learning_rate_gates'])

    # optimal gates
    train_tracker = Tracker()
    eval_tracker = Tracker()
    train_tracker.root_gate_init = deepcopy(model_tree.root)
    train_tracker.leaf_gate_init = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])

    for epoch in range(hparams['n_epoch']):
        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]

        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()
            output = model_tree([x_train[j] for j in idx_batch], y_train[idx_batch])
            loss = output['loss']
            loss.backward()
            if (len(x_train) // hparams['batch_size'] * epoch + i) % hparams['n_mini_batch_update_gates'] == 0:
                print("optimizing gates...")
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

        # print every n_batch_print mini-batches
        if epoch % hparams['n_epoch_eval'] == 0:
            # stats on train
            train_tracker.update(model_tree, model_tree(input.x_train, input.y_train), input.y_train, epoch, i)
            eval_tracker.update(model_tree, model_tree(input.x_eval, input.y_eval), input.y_eval, epoch, i)

            # compute
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (
                epoch, i, train_tracker.loss[-1], eval_tracker.loss[-1]))
            print('[Epoch %d, batch %d] training, eval ref_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.ref_reg_loss[-1], eval_tracker.ref_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval size_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.size_reg_loss[-1], eval_tracker.size_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (
                epoch, i, train_tracker.acc[-1], eval_tracker.acc[-1]))

    print("Running time for training %d epoch: %.3f seconds" % (hparams['n_epoch'], time.time() - start))
    print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
              eval_tracker.n_iter_opt[0],
              eval_tracker.n_iter_opt[1],))

    return model_tree, train_tracker, eval_tracker, time.time() - start


def run_output(model_tree, dafi_tree, hparams, input, train_tracker, eval_tracker, run_time):
    y_pred_train = (model_tree(input.x_train, input.y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_eval = (model_tree(input.x_eval, input.y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred = (model_tree(input.x, input.y)['y_pred'].detach().numpy() > 0.5) * 1.0
    train_accuracy = sum(y_pred_train == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy = sum(y_pred_eval == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy = sum(y_pred == input.y.numpy()) * 1.0 / len(input.x)
    train_auc = roc_auc_score(input.y_train.numpy(), y_pred_train, average='macro')
    eval_auc = roc_auc_score(input.y_eval.numpy(), y_pred_eval, average='macro')
    overall_auc = roc_auc_score(input.y.numpy(), y_pred, average='macro')
    train_brier_score = brier_score_loss(input.y_train.numpy(), y_pred_train)
    eval_brier_score = brier_score_loss(input.y_eval.numpy(), y_pred_eval)
    overall_brier_score = brier_score_loss(input.y.numpy(), y_pred)

    y_pred_train_dafi = (dafi_tree(input.x_train, input.y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_eval_dafi = (dafi_tree(input.x_eval, input.y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_dafi = (dafi_tree(input.x, input.y)['y_pred'].detach().numpy() > 0.5) * 1.0
    train_accuracy_dafi = sum(y_pred_train_dafi == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy_dafi = sum(y_pred_eval_dafi == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy_dafi = sum(y_pred_dafi == input.y.numpy()) * 1.0 / len(input.x)
    train_auc_dafi = roc_auc_score(input.y_train.numpy(), y_pred_train_dafi, average='macro')
    eval_auc_dafi = roc_auc_score(input.y_eval.numpy(), y_pred_eval_dafi, average='macro')
    overall_auc_dafi = roc_auc_score(input.y.numpy(), y_pred_dafi, average='macro')
    train_brier_score_dafi = brier_score_loss(input.y_train.numpy(), y_pred_train_dafi)
    eval_brier_score_dafi = brier_score_loss(input.y_eval.numpy(), y_pred_eval_dafi)
    overall_brier_score_dafi = brier_score_loss(input.y.numpy(), y_pred_dafi)

    with open('../output/%s/results_cll_4D.csv' % hparams['experiment_name'], "a+") as file:
        file.write(
            "%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f([%d; %d]), %.3f([%d; %d]), %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f\n" % (
                hparams['random_state'],
                train_accuracy, eval_accuracy, overall_accuracy,
                train_accuracy_dafi, eval_accuracy_dafi, overall_accuracy_dafi,
                train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1],
                eval_tracker.acc_opt, eval_tracker.n_iter_opt[0], eval_tracker.n_iter_opt[1],
                model_tree(input.x_train, input.y_train)['log_loss'].detach().numpy(),
                model_tree(input.x_eval, input.y_eval)['log_loss'].detach().numpy(),
                model_tree(input.x, input.y)['log_loss'].detach().numpy(),
                dafi_tree(input.x_train, input.y_train)['log_loss'].detach().numpy(),
                dafi_tree(input.x_eval, input.y_eval)['log_loss'].detach().numpy(),
                dafi_tree(input.x, input.y)['log_loss'].detach().numpy(),
                train_auc, eval_auc, overall_auc, train_auc_dafi, eval_auc_dafi, overall_auc_dafi,
                train_brier_score, eval_brier_score, overall_brier_score,
                train_brier_score_dafi, eval_brier_score_dafi, overall_brier_score_dafi,
                run_time
            ))

    return {
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "overall_accuracy": overall_accuracy,
        "train_accuracy_dafi": train_accuracy_dafi,
        "eval_accuracy_dafi": eval_accuracy_dafi,
        "overall_accuracy_dafi": overall_accuracy_dafi,
        "train_auc": train_auc,
        "eval_auc": eval_auc,
        "overall_auc": overall_auc,
        "train_auc_dafi": train_auc_dafi,
        "eval_auc_dafi": eval_auc_dafi,
        "overall_auc_dafi": overall_auc_dafi,
        "train_brier_score": train_brier_score,
        "eval_brier_score": eval_brier_score,
        "overall_brier_score": overall_brier_score,
        "train_brier_score_dafi": train_brier_score_dafi,
        "eval_brier_score_dafi": eval_brier_score_dafi,
        "overall_brier_score_dafi": overall_brier_score_dafi,

    }


def run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, input, output_metric_dict):
    x_range = [i * hparams['n_epoch_eval'] for i in range(hparams['n_epoch'] // hparams['n_epoch_eval'])]
    filename_metric = "../output/%s/metrics.png" % (hparams['experiment_name'])
    util_plot.plot_metrics(x_range, train_tracker, eval_tracker, filename_metric,
                           dafi_tree(input.x_train, input.y_train),
                           dafi_tree(input.x_eval, input.y_eval),
                           output_metric_dict)


def run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, input):
    filename_root_pas = "../output/%s/root_pos.png" % (hparams['experiment_name'])
    filename_root_neg = "../output/%s/root_neg.png" % (hparams['experiment_name'])
    filename_leaf_pas = "../output/%s/leaf_pos.png" % (hparams['experiment_name'])
    filename_leaf_neg = "../output/%s/leaf_neg.png" % (hparams['experiment_name'])

    ####### compute model_pred_prob
    model_pred_prob = model_tree(input.x, input.y)['y_pred'].detach().numpy()
    model_pred = (model_pred_prob > 0.5) * 1.0
    dafi_pred_prob = dafi_tree(input.x, input.y)['y_pred'].detach().numpy()
    dafi_pred = (dafi_pred_prob > 0.5) * 1.0

    # filter out samples according DAFI gate at root for visualization at leaf
    filtered_normalized_x = [dh.filter_rectangle(x, 0, 1, 0.402, 0.955, 0.549, 0.99) for x in input.x]
    util_plot.plot_cll(input.x, filtered_normalized_x, input.y, input.features, model_tree, input.reference_tree,
                       train_tracker, eval_tracker, model_pred, model_pred_prob, dafi_pred, dafi_pred_prob,
                       filename_root_pas, filename_root_neg, filename_leaf_pas, filename_leaf_neg)