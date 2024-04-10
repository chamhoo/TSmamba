import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from .star import STAR
from .utils import *

import json
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



class processor(object):
    def __init__(self, args, model_parameters):
        # initialization
        self.args = args
        self.model_parameters = model_parameters
        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args, model_parameters)
        # eval parameters
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")
        self.set_optimizer()
        
        # if cuda
        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1
        self.scheduler = None


    def save_model(self, epoch, train_loss):

        model_path = self.args.save_dir + '/' + self.args.modelname + '/' + self.args.modelname + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)


    def load_model(self):
        start_epoch = 0
        if self.args.load_model_id is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.modelname + '/' + self.args.modelname + '_' + \
                                        str(self.args.load_model_id) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                # 假设 optimizer 是你的优化器实例
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # epoch and loss
                start_epoch = model_epoch + 1
                print('Loaded checkpoint at epoch', model_epoch)
        return start_epoch


    def set_optimizer(self):
        lr = self.args.learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')


    def test(self):
        print('Testing begin')
        self.load_model()
        self.net.eval()
        test_error, test_final_error = self.test_epoch()
        print("&&&&&&& Hyper Parameters &&&&&&&&&&&&")
        for key, value in self.model_parameters.items():
            print(f"{key}: {value}")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"Test Set: {self.args.test_set}")
        print(f"ADE: {test_error}")
        print(f"FDE: {test_final_error}")


    def train(self):
        # scheduler_method 
        # batch_around_ped 
        # num_epochs 
        # test_set 
        # learning_rate 
        # early_stopping 
        print('Training begin ...')
        print("&&&&&&& Hyper Parameters &&&&&&&&&&&&")
        for key, value in self.model_parameters.items():
            print(f"{key}: {value}")
        print("")
        print(f"Scheduler Method: {self.args.scheduler_method}")
        print(f"batch_around_ped: {self.args.batch_around_ped}")
        print(f"# of epochs: {self.args.num_epochs}")
        print(f"Test Set: {self.args.test_set}")
        print(f"Learning Rate: {self.args.learning_rate}")
        print(f"Early Stopping: {self.args.early_stop}")
        print(f"Model Path: {self.args.save_base_dir}")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        test_error, test_final_error = 0, 0
        start_epoch = self.load_model()
        # 定义 LR
        if self.args.scheduler_method == "OneCycleLR":
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=0.01, 
                steps_per_epoch=self.dataloader.trainbatchnums, 
                epochs=self.args.num_epochs - start_epoch
                )
            
        elif self.args.scheduler_method == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1, 
                patience=self.args.patience//2, 
                )
        else:
            pass
        
        for epoch in range(start_epoch, self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error = self.test_epoch()
                # Best ADE & FDE
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
                # if best epoch:
                if (test_final_error <= self.best_fde) or (test_error <= self.best_ade):
                    self.save_model(epoch, train_loss)
                    self.best_epoch = epoch
                else:
                    if (self.best_epoch + self.args.patience < epoch) and self.args.early_stop:
                        break

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 1 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch >= self.args.start_test:
                print(
                    '----epoch {}, lr={:.6f}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, current_lr, train_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}, lr={:.6f}'
                      .format(epoch, train_loss, current_lr))
            # 更新学习率
            if self.args.scheduler_method == "ReduceLROnPlateau":
                self.scheduler.step(test_final_error)

    def train_epoch(self, epoch):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0
        # 初始化进度条
        pbar = tqdm(total=self.dataloader.trainbatchnums, leave=False)

        for batch in range(self.dataloader.trainbatchnums):
            # self.optimizer.zero_grad()
            # start = time.time()
            inputs_ori, batch_id = self.dataloader.get_train_batch(batch)
            inputs = []
            for idx, contents in enumerate(inputs_ori):
                if idx != len(inputs_ori) - 2:
                    contents = torch.Tensor(contents).cuda()
                inputs.append(contents)
            # inputs = tuple([torch.Tensor(i) for i in inputs])
            # inputs = tuple([i.cuda() for i in inputs])
            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, scenes, batch_pednum = inputs
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], scenes, batch_pednum

            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss = loss + (torch.sum(loss_o * lossmask / num))
            loss_epoch = loss_epoch + loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)

            self.optimizer.step()

            # end = time.time()
            if self.args.scheduler_method == "OneCycleLR":
                self.scheduler.step()
            # if batch % self.args.show_step == 0 and self.args.ifshow_detail:
            #     print(
            #         'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
            #                                                                                    self.dataloader.trainbatchnums,
            #                                                                                    epoch, loss.item(),
            #                                                                                    end - start))
            # 更新进度条并显示当前batch的loss
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
        pbar.close()
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch

    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in range(self.dataloader.testbatchnums):

            inputs_ori, batch_id = self.dataloader.get_test_batch(batch)
            inputs = []
            for idx, contents in enumerate(inputs_ori):
                if idx != len(inputs_ori) - 2:
                    contents = torch.Tensor(contents)
                    if self.args.using_cuda:
                        contents = contents.cuda()
                inputs.append(contents)

            batch_abs, batch_norm, shift_value, seq_list, scenes, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], scenes, batch_pednum

            all_output = []
            for i in range(self.args.sample_num):
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :2],
                                                                        self.args.obs_length, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
