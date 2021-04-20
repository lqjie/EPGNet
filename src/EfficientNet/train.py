import warnings
import torch
from utils import *
import os
import time
from datetime import datetime
from cosine import CosineAnnealingWarmUpRestarts
warnings.filterwarnings("ignore")

class Fitter:
    
    def __init__(self, model, device, config, n_iter_per_ep=2500):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./ckpt/{config["exp_name"]}'

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5
        self.best_summary_acc = 0

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["LR"])
        
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=config["n_epochs"]*n_iter_per_ep,
            T_mult=1,
            eta_max=config["eta_max"],
            T_up=config["T_up"]*n_iter_per_ep,
        )
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader, test_loader):
        CheckPointName_Output = 'model'
        for e in range(self.config['n_epochs']):
            if self.config["verbose"]:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]:Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.4f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/{CheckPointName_Output}-last.bin')

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)
            self.log(f'[RESULT]:  Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.4f}, time: {(time.time() - t):.5f}')

            # t = time.time()
            # summary_loss_test, final_scores_test = self.validation(test_loader)
            # self.log(f'[RESULT] : TEST Epoch: {self.epoch}, summary_loss: {summary_loss_test.avg:.5f}, final_score: {final_scores_test.avg:.5f}, time: {(time.time() - t):.5f}')

            if self.epoch == 0:
                self.best_summary_loss = summary_loss.avg
                self.best_summary_acc = final_scores.avg
                self.model.eval()
            else:
                if summary_loss.avg < self.best_summary_loss: 
                    print('[BEST LOSS]: --> LOWEST loss achieved :: Saving checkpoint <--')
                    self.best_summary_loss = summary_loss.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/{CheckPointName_Output}-best.bin')
                if final_scores.avg >= self.best_summary_acc: 
                    print('[BEST -Acc-]: --> HIGHEST -Acc- achieved :: Saving checkpoint <--')
                    self.best_summary_acc = final_scores.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/{CheckPointName_Output}-bestAcc.bin')

            if self.config["validation_scheduler"]:
                self.scheduler.step()

            self.epoch += 1
    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = AccMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config["verbose"]:
                if step % self.config["verbose_step"] == 0:
                    print(
                        f'Epoch = {self.epoch}' + \
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.4f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores
    def test(self, test_loader):
        t = time.time()
        summary_loss_FINALTEST, final_scores_FINALTEST = self.validation(test_loader)
        self.log(f'[RESULT]: TEST --> summary_loss: {summary_loss_FINALTEST.avg:.5f}, final_score: {final_scores_FINALTEST.avg:.4f}, time: {(time.time() - t):.5f} <--')

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = AccMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config["verbose"]:
                if step % self.config["verbose_step"] == 0:
                    print(
                        f'Epoch = {self.epoch}' + \
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.getVal:.5f}, final_score: {final_scores.getVal:.4f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config["step_scheduler"]:
                self.scheduler.step()

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#----- USE THIS IF WANT TO USE ALL MODEL HYPER-PARAMETERS (NOT ONLY THE WEIGHTS -----
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#        self.best_summary_loss = checkpoint['best_summary_loss']
#        self.epoch = checkpoint['epoch'] + 1
#------------------------------------------------------------------------------------------------------------------------------------------------------#

    def log(self, message):
        if self.config["verbose"]:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

