import copy
import time
import torch
from scipy.stats import spearmanr, pearsonr
from dataset import ImageDataset as dataset
from NestedNet import Model as model
from util import Context
from sklearn.metrics import mean_squared_error
from math import sqrt



config = Context().get_config()
logger = Context().get_logger()

# model
MODEL_SAVE_PATH = config["project"]["save_model"]
MODEL_LOAD_PATH = config["project"]["load_model"]

# datasetScores
INPUT_PATH = config["dataset"]["image_dir"]
LIST_SCORE = config["dataset"]["score_file"]

# train para
LEARNING_RATE = config['train']['learning_rate']
NUM_EPOCHS = config['train']["num_epochs"]
MUTI_GPU_MODE = config["train"]["muti_gpu"]
WEIGHT_DECAY = config["train"]["weight_decay"]



def train_model(model, device, optimizer, dataloaders, scheduler, num_epochs=100):

    since = time.time()
    best_PLCC = -1.0
    best_SRCC = -1.0

    srcc_set = []
    srcc_test_set = []
    plcc_set = []
    plcc_test_set = []
    for epoch in range(num_epochs):
        logger.critical('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            groundtruth_mos_set = []
            predict_mos_set = []
            epoch_phase_loss = 0.0
            epoch_phase_size = 0

            # Iterate over data.
            for batch_index, (
            r_all_patch_set, d_all_patch_set, mos_set) in enumerate(
                    dataloaders[phase]):
                r_all_patch_set = r_all_patch_set.to(device)
                d_all_patch_set = d_all_patch_set.to(device)
                mos_set = mos_set.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    total_loss, predict_mos = model(r_all_patch_set, d_all_patch_set, mos_set)



                    total_loss = total_loss.float()
                    predict_mos = predict_mos.reshape(mos_set.shape).float()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                    else:
                        predict_mos=torch.mean(predict_mos)
                        mos_set=torch.mean(mos_set)

                # statistics
                current_average_loss = total_loss.item()
                epoch_phase_loss += current_average_loss * r_all_patch_set.size(0)
                epoch_phase_size += r_all_patch_set.size(0)

                groundtruth_mos_set.append(mos_set.flatten())
                predict_mos_set.append(predict_mos.flatten())
                logger.info('batch {} Loss: {:.4f} '.format(batch_index, current_average_loss))

            epoch_average_loss = epoch_phase_loss / epoch_phase_size
            groundtruth_mos_set = torch.cat(groundtruth_mos_set).flatten().data.cpu().numpy()
            predict_mos_set = torch.cat(predict_mos_set).flatten().data.cpu().numpy()
            epoch_PLCC = pearsonr(groundtruth_mos_set, predict_mos_set)[0]  # (corr,p value)
            epoch_SRCC = spearmanr(groundtruth_mos_set, predict_mos_set)[0]  # (corr,p value)
            epoch_RMSE = sqrt(mean_squared_error(predict_mos_set, groundtruth_mos_set))
            logger.critical('epoch: {} {} Loss: {:.4f} PLCC: {:.4f} SRCC: {:.4f} RMSE: {:.4}'.format(epoch,
                                                                                                     phase,
                                                                                                     epoch_average_loss,
                                                                                                     epoch_PLCC,
                                                                                                     epoch_SRCC,
                                                                                                     epoch_RMSE))

            # save every 5 epoch
            if epoch % 5 == 0 and phase == "test":
                CUR_MODEL_SAVE_PATH = '{pt}_{epoch}'.format(pt=MODEL_SAVE_PATH, epoch=epoch)
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, CUR_MODEL_SAVE_PATH)

            if phase == 'test' and epoch_SRCC > best_SRCC:
                best_SRCC = epoch_SRCC
                best_PLCC = epoch_PLCC
                CUR_MODEL_SAVE_PATH = '{pt}_best_srcc'.format(pt=MODEL_SAVE_PATH)
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, CUR_MODEL_SAVE_PATH)

            # record plcc and srcc for drawing
            if phase == 'train':
                scheduler.step(epoch_average_loss)
                plcc_set.append(epoch_PLCC)
                srcc_set.append(epoch_SRCC)
            if phase == 'test':
                plcc_test_set.append(epoch_PLCC)
                srcc_test_set.append(epoch_SRCC)

        logger.info('-' * 10)
        logger.info('Epoch {}/{} done \n'.format(epoch, num_epochs - 1))  # epoch end

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best  PLCC: {:4f}  Best  SRCC: {:4f}'.format(best_PLCC, best_SRCC))
    return model, plcc_set, plcc_test_set, srcc_set, srcc_test_set


if __name__ == '__main__':
    video_datasets = {x: dataset(LIST_SCORE, INPUT_PATH, mode=x) for x in ['train', 'test']}
    dataloaders={'train':torch.utils.data.DataLoader(video_datasets['train'], batch_size=40, shuffle=True, num_workers=4),
                 'test':torch.utils.data.DataLoader(video_datasets['test'], batch_size=40, shuffle=False, num_workers=4)
                 }


    # model
    model = model()

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.device_count() > 1 and MUTI_GPU_MODE == True:
        # use all gpus
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
        logger.info("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        model = model.to(device)

    # load existed model
    if (MODEL_LOAD_PATH != None):
        logger.info("load model in {}".format(MODEL_LOAD_PATH))
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(MODEL_LOAD_PATH).items()})
        model = model.to(device)

    logger.info("learning rate: {}".format(LEARNING_RATE))

    res_params = list(map(id, model.resnet.parameters()))
    other_params = filter(lambda p: id(p) not in res_params,
                         model.parameters())
    res_params_ = filter(lambda p: id(p)  in res_params,
                         model.parameters())
    optimizer = torch.optim.Adam([ {'params': res_params_},{'params': other_params,'lr':LEARNING_RATE*10}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_model_wts, plcc_set, plcc_test, srcc_set, srcc_test = train_model(model, device, optimizer, dataloaders,
                                                                           scheduler, num_epochs=NUM_EPOCHS)
