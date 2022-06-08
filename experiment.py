import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import time
from dataloaders import data_dict,data_set
from sklearn.metrics import confusion_matrix
# import models
from models.crossatten.model import Cross_TS,TSTransformer_Basic
from models.deepconvlstm import DeepConvLSTM
from models.SA_HAR import SA_HAR
from models.deepconvlstm_attn import DeepConvLSTM_ATTN
from models.Attend import AttendDiscriminate
from models.Attend_new import AttendDiscriminate_new
from models.CNN_freq import CNN_Freq_Model
from models.CNN_LSTM_FREQ import CNN_LSTM_FREQ_Model
from models.CNN_LSTM_TIME import CNN_LSTM_TIME_Model
from models.CNN_LSTM_TIME_FREQ import CNN_LSTM_CROSS_Model
from models.CFC import CFC_Model
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from utils import EarlyStopping, adjust_learning_rate_class, mixup_data, MixUpLoss
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
#torch.manual_seed(0)
#import random
#random.seed(0)
#np.random.seed(0)


class Exp(object):
    def __init__(self, args):
        self.args = args
        # set the device
        self.device = self.acquire_device()
        self.optimizer_dict = {"Adam":optim.Adam}
        self.criterion_dict = {"MSE":nn.MSELoss,"CrossEntropy":nn.CrossEntropyLoss}

        self.model  = self.build_model().to(self.device)
        print("Done!")
        print("Parameter :", np.sum([para.numel() for para in self.model.parameters()]))


    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def build_model(self):
        if self.args.model_type in ["time","freq","cross"]:
            model  = Cross_TS(self.args)
            print("Build the conv_TS model!")
        elif self.args.model_type == "basic":
            model  = TSTransformer_Basic(self.args)
            print("Build the basic TS model!")
        elif self.args.model_type == "deepconvlstm":
            model  = DeepConvLSTM(self.args.c_in, self.args.num_classes)
            print("Build the DeepConvLSTM model!")
        elif self.args.model_type == "sahar":
            model  = SA_HAR(self.args.c_in, self.args.input_length, self.args.num_classes)
            print("Build the SA_HAR model!")
        elif self.args.model_type == "deepconvlstm_attn":
            model  = DeepConvLSTM_ATTN(self.args.c_in, self.args.num_classes)
            print("Build the deepconvlstm_attn model!")
        elif self.args.model_type == "attend":
            model  = AttendDiscriminate(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate model!")
        elif self.args.model_type == "attend_new":
            model  = AttendDiscriminate_new(self.args.c_in, self.args.num_classes)
            print("Build the AttendDiscriminate_new model!")
        elif self.args.model_type == "cnn_freq":
            model  = CNN_Freq_Model((1,self.args.c_in, self.args.sampling_freq, self.args.input_length ), self.args.num_classes)
            print("Build the CNN_Freq_Model model!")		
        elif self.args.model_type == "cnn_lstm_freq":
            model  = CNN_LSTM_FREQ_Model((self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_FREQ_Model model!")					
        elif self.args.model_type == "cnn_lstm_time":
            model  = CNN_LSTM_TIME_Model((self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CNN_LSTM_TIME_Model model!")		
        elif self.args.model_type == "cnn_lstm_cross":
            model  = CNN_LSTM_CROSS_Model((self.args.input_length, self.args.c_in ),(self.args.input_length,self.args.sampling_freq ), self.args.num_classes)
            print("Build the CNN_LSTM_CROSS_Model model!")
        elif self.args.model_type == "cfc":
            model  = CFC_Model((1,self.args.input_length, self.args.c_in ), self.args.num_classes)
            print("Build the CFC model!")			
        else:
            raise NotImplementedError
        return model.double()

    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    def _get_data(self, data, flag="train", weighted_sampler = False):
        if flag == 'train':
            shuffle_flag = True # ++++++++++++++++++++++++++++
        else:
            shuffle_flag = False

        data  = data_set(self.args,data,flag)
        if weighted_sampler and flag == 'train':

            sampler = WeightedRandomSampler(
                data.act_weights, len(data.act_weights)
            )

            data_loader = DataLoader(data, 
                                     batch_size   =  self.args.batch_size,
                                     #shuffle      =  shuffle_flag,
                                     num_workers  =  0,
                                     sampler=sampler,
                                     drop_last    =  False)
        else:
            data_loader = DataLoader(data, 
                                     batch_size   =  self.args.batch_size,
                                     shuffle      =  shuffle_flag,
                                     num_workers  =  0,
                                     drop_last    =  False)
        return data_loader

    def train(self):
        # save_path_need ++++++++++++
        dateTimeObj = datetime.now()
        setting = "Time_{}_{}_{}_{}_data{}_mode{}_model{}_win{}_mask{}_drop{}_depth{}_dmodel{}_Weightsample{}".format(dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour, dateTimeObj.minute,
                                                                                                                      self.args.data_name, self.args.exp_mode, self.args.model_type, 
                                                                                                                      self.args.windowsize, self.args.attention_layer_types,
                                                                                                                      self.args.drop_transition, self.args.cross_depth, 
                                                                                                                      self.args.token_d_model,self.args.weighted_sampler)
        path = os.path.join(self.args.to_save_path,'logs/'+setting)
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        # load the data
        dataset = data_dict[self.args.data_name](self.args)
        file_name = "Time_{}_{}_{}_{}_data{}_mode{}_model{}_win{}_mask{}_drop{}_depth{}_dmodel{}_Weightsample{}.txt".format(dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour, dateTimeObj.minute,
                                                                                                                            self.args.data_name, self.args.exp_mode, self.args.model_type, 
                                                                                                                            self.args.windowsize, self.args.attention_layer_types, 
                                                                                                                            self.args.drop_transition, self.args.cross_depth, 
                                                                                                                            self.args.token_d_model,self.args.weighted_sampler)


        if self.args.to_save_path is not None:
            file_name = os.path.join(self.args.to_save_path, file_name)
        log = open(file_name, "w+")
		
		
		

        if self.args.exp_mode in ["LOCV", "SOCV","Given"]:
            file_name_1 = "Time_{}_{}_{}_{}_data{}_mode{}_model{}_win{}_mask{}_score.txt".format(dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour, dateTimeObj.minute,
                                                                                          self.args.data_name, self.args.exp_mode, self.args.model_type, self.args.windowsize, self.args.attention_layer_types)
            file_name_1 = os.path.join(self.args.to_save_path, file_name_1)
            #log_1 = open(file_name_1, "w+")




        print("================ {} Mode ====================".format(dataset.exp_mode))
        print("================ {} CV ======================".format(dataset.num_of_cv))
        log.write("================ {} Mode ====================".format(dataset.exp_mode))
        log.write("\n")
        log.write("================ {} CV ======================".format(dataset.num_of_cv))
        log.write("\n")
        num_of_cv = dataset.num_of_cv

        log.close()
        for iter in range(num_of_cv):
            log = open(file_name, "a")
            log_1 = open(file_name_1, "a")

            cv_path = os.path.join(path,"cv_{}".format(iter))
            if not os.path.exists(cv_path):
                os.makedirs(cv_path)
            print("================ Build the model ================ ".format(iter+1))	
            if self.args.mixup:
                print(" Using Mixup Training")				
            self.model  = self.build_model().to(self.device)

            print("================ the {} th CV Experiment ================ ".format(iter+1))
            log.write("================ the {} th CV Experiment ================ ".format(iter+1))
            log.write("\n")
            log.write("               NEW BEGIN                  \n")
            log.write("\n")
            dataset.update_train_val_test_keys()

            if self.args.exp_mode in ["LOCV","SOCV"] and self.args.cv_skip_number is not None:
                index_of_cv = dataset.index_of_cv
                if index_of_cv<=self.args.cv_skip_number:
                    print("================Skip the {} CV Experiment================".format(index_of_cv+1))
                    log.write("================ Skip the {} CV Experiment================ ".format(index_of_cv+1))
                    log.write("\n")
                    continue
            #print("After update the train test split , the class weight :" , dataset.act_weights)
            # get the loader of train val test
            train_loader = self._get_data(dataset, flag = 'train', weighted_sampler = self.args.weighted_sampler )
            val_loader = self._get_data(dataset, flag = 'vali', weighted_sampler = self.args.weighted_sampler)
            test_loader   = self._get_data(dataset, flag = 'test', weighted_sampler = self.args.weighted_sampler)
            #class_weights=torch.tensor(dataset.act_weights,dtype=torch.double).to(self.device)
            train_steps = len(train_loader)

            early_stopping        = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
            learning_rate_adapter = adjust_learning_rate_class(self.args,True)

            model_optim = self._select_optimizer()
            #if self.args.weighted == True:
            #    criterion =  nn.CrossEntropyLoss(reduction="mean",weight=class_weights).to(self.device)#self._select_criterion()
            #else:
            #    criterion =  nn.CrossEntropyLoss(reduction="mean").to(self.device)#self._select_criterion()
            criterion =  nn.CrossEntropyLoss(reduction="mean").to(self.device)
            val_loss_min = np.Inf
            filnal_test_f_w = 0
            filnal_test_f_m = 0
			
			
			
            for epoch in range(self.args.train_epochs):
                train_loss = []
                #preds = []
                #trues = []
                self.model.train()

                epoch_time = time.time()
                #print(".....")
                for i, (batch_x1,batch_x2,batch_y) in enumerate(train_loader):

                    model_optim.zero_grad()

                    if "cross" in self.args.model_type:
                        batch_x1 = batch_x1.double().to(self.device)
                        batch_x2 = batch_x2.double().to(self.device)
                        batch_y = batch_y.long().to(self.device)
                        # model prediction
                        if self.args.output_attention:
                            outputs = self.model(batch_x1,batch_x2)[0]
                        else:
                            outputs = self.model(batch_x1,batch_x2)
                    else:
                        batch_x1 = batch_x1.double().to(self.device)
                        batch_y = batch_y.long().to(self.device)

                        if self.args.mixup:
                            batch_x1, batch_y = mixup_data(batch_x1, batch_y, self.args.alpha)

                        # model prediction
                        if self.args.output_attention:
                            outputs = self.model(batch_x1)[0]
                        else:
                            outputs = self.model(batch_x1)

                    if self.args.mixup:
                        criterion = MixUpLoss(criterion)
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()

                    #preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
                    #trues.extend(list(batch_y.detach().cpu().numpy()))   

                print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                log.write("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
                log.write("\n")
                train_loss = np.average(train_loss)
                #train_acc_1 = accuracy_score(preds,trues)
                vali_loss , vali_acc, vali_f_w,  vali_f_macro,  vali_f_micro = self.validation(val_loader, criterion)
                #test_loss , test_acc, test_f_w,  test_f_macro,  test_f_micro = self.validation(test_loader, criterion)
                #_         , train_acc,       _,         _ = self.validation(train_loader, criterion)

                #print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Accuracy {3:.7f} Vali Loss: {4:.7f} Vali Accuracy: {5:.7f}  Vali weighted F1: {6:.7f}  Vali macro F1 {7:.7f}".format(
                #    epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc, vali_f_w, vali_f_m))
                print("VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))
                #print("TEST: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Test Loss: {3:.7f} Test Accuracy: {4:.7f}  Test weighted F1: {5:.7f}  Test macro F1 {6:.7f} ".format(
                #    epoch + 1, train_steps, train_loss, test_loss, test_acc, test_f_w, test_f_macro))
                log.write("VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} \n".format(
                    epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))
                #log.write("TEST: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Test Loss: {3:.7f} Test Accuracy: {4:.7f}  Test weighted F1: {5:.7f}  Test macro F1 {6:.7f} \n".format(
                #    epoch + 1, train_steps, train_loss, test_loss, test_acc, test_f_w, test_f_macro))

                #if vali_loss<=val_loss_min:
                #    val_loss_min = vali_loss
                #    filnal_test_f_m = test_f_macro
                #    filnal_test_f_w = test_f_w

                early_stopping(vali_loss, self.model, cv_path, vali_f_macro, vali_f_w, log)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                log.write("----------------------------------------------------------------------------------------\n")
                log.flush()
                learning_rate_adapter(model_optim,vali_loss)

            #self.model  = self.build_model().to(self.device)
			
            print("Loading the best validation model!")
            self.model.load_state_dict(torch.load(cv_path+'/'+'best_vali.pth'))
            #model.eval()
            test_loss , test_acc, test_f_w,  test_f_macro,  test_f_micro = self.validation(test_loader, criterion, iter+1)
            print("Final Test Performance : Test Accuracy: {0:.7f}  Test weighted F1: {1:.7f}  Test macro F1 {2:.7f} ".format (test_acc, test_f_w, test_f_macro))
            log.write("Final Test Performance : Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n\n\n\n\n\n\n\n".format(test_f_w, test_f_macro))
            log.flush()

            log_1.write("Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n".format(test_f_w, test_f_macro))
            log_1.flush()
            #torch.save(self.model.state_dict(), os.path.join(cv_path,'last.pth'))
            log.close()
            log_1.close()
            #best_model_path = cv_path+'/'+'checkpoint.pth'
            #self.model.load_state_dict(torch.load(best_model_path))

    def prediction_test(self):
        assert self.args.exp_mode == "Given"
        model = self.build_model().to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.path,'cv_0/best_vali.pth')))
        model.eval()
        dataset = data_dict[self.args.data_name](self.args)
        dataset.update_train_val_test_keys()
        test_loader   = self._get_data(dataset, flag = 'test')
        preds = []
        trues = []
        for i, (batch_x1,batch_x2,batch_y) in enumerate(test_loader):
            if "cross" in self.args.model_type:
                batch_x1 = batch_x1.double().to(self.device)
                batch_x2 = batch_x2.double().to(self.device)
                batch_y = batch_y.long().to(self.device)
                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x1,batch_x2)[0]
                else:
                    outputs = self.model(batch_x1,batch_x2)
            else:
                batch_x1 = batch_x1.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # model prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x1)[0]
                else:
                    outputs = self.model(batch_x1)

            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy())) 
		
        acc = accuracy_score(preds,trues)
        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')

        return preds,trues



    def validation(self, data_loader, criterion, index_of_cv=None):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x1,batch_x2,batch_y) in enumerate(data_loader):

                if "cross" in self.args.model_type:
                    batch_x1 = batch_x1.double().to(self.device)
                    batch_x2 = batch_x2.double().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    # model prediction
                    if self.args.output_attention:
                        outputs = self.model(batch_x1,batch_x2)[0]
                    else:
                        outputs = self.model(batch_x1,batch_x2)
                else:
                    batch_x1 = batch_x1.double().to(self.device)
                    batch_y = batch_y.long().to(self.device)

                    # model prediction
                    if self.args.output_attention:
                        outputs = self.model(batch_x1)[0]
                    else:
                        outputs = self.model(batch_x1)


                pred = outputs.detach()#.cpu()
                true = batch_y.detach()#.cpu()

                loss = criterion(pred, true) 
                total_loss.append(loss.cpu())
				
                preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
                trues.extend(list(batch_y.detach().cpu().numpy()))   
				
        total_loss = np.average(total_loss)
        acc = accuracy_score(preds,trues)
        #f_1 = f1_score(trues, preds)
        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')
        if index_of_cv:
            cf_matrix = confusion_matrix(trues, preds)
            with open("{}.npy".format(index_of_cv), 'wb') as f:
                np.save(f, cf_matrix)
            plt.figure()
            sns.heatmap(cf_matrix, annot=True)
            plt.savefig("{}.png".format(index_of_cv))
        self.model.train()

        return total_loss,  acc, f_w,  f_macro, f_micro#, f_1


