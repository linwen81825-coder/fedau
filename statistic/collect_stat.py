
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class CollectStatistics:

    def __init__(self, results_eval_file_prefix=os.path.dirname(__file__) + '/results'):

        self.results_eval_file_name = results_eval_file_prefix + '.csv'

        # 记录每个 seed 的历史最佳测试准确率
        self.best_accuracy_by_seed = {}

        with open(self.results_eval_file_name, 'a') as f:
            f.write(
                'sim_seed,num_iter,training_loss,training_accuracy,'
                'test_loss,test_accuracy,best_accuracy\n'
            )

    def collect_stat_eval(self, seed, num_iter, model,
                          train_data_loader, test_data_loader, w_global=None):

        # 全局模型在训练集上的 Loss 和 Acc
        train_loss, train_accuracy = model.accuracy(
            train_data_loader,
            w_global,
            device,
            transform=transform_train_eval
        )

        # 全局模型在测试集上的 Loss 和 Acc
        test_loss, test_accuracy = model.accuracy(
            test_data_loader,
            w_global,
            device
        )

        # 更新历史最佳测试准确率
        best_accuracy = max(
            self.best_accuracy_by_seed.get(seed, 0.0),
            test_accuracy
        )

        self.best_accuracy_by_seed[seed] = best_accuracy

        # 控制台输出
        print(
            f"Round {num_iter // iters_per_round:5d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy * 100:.2f}% | "
            f"Best Acc: {best_accuracy * 100:.2f}%",
            flush=True
        )

        # 保存 CSV
        with open(self.results_eval_file_name, 'a') as f:
            f.write(
                f'{seed},{num_iter},{train_loss},{train_accuracy},'
                f'{test_loss},{test_accuracy},{best_accuracy}\n'
            )
