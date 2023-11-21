# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, loss_function, confidence_treshold=0, iou_treshold=0,autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.confidence_treshold = confidence_treshold
        self.iou_treshold = iou_treshold
        self.loss_function = loss_function
        
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.mse = nn.MSELoss(reduction='mean')

    def __call__(self, p, targets, mode='train'):

        #print(f'conf = {self.confidence_treshold} iou = {self.iou_treshold} lf = {self.loss_function}')

        if mode == 'train':
            if self.loss_function == 'ordinary':
                return self.ordinary_loss(p, targets)
            elif self.loss_function == 'ignore':
                decent_predictions = self.screen_time_estimation_loss_ignore_good_detections(p, targets)
                ordinary_loss, ordinary_separate_losses = self.ordinary_loss(p, targets, decent_predictions=decent_predictions)
                
                return ordinary_loss, ordinary_separate_losses

            elif self.loss_function == 'motivate':

                motivate_loss, motivate_separate_losses, decent_predictions = self.screen_time_estimation_loss_motivate_good_detections(p, targets)
                ordinary_loss, ordinary_separate_losses = self.ordinary_loss(p, targets, decent_predictions=decent_predictions)
                total_loss = motivate_loss + ordinary_loss
                separate_losses = motivate_separate_losses + ordinary_separate_losses
                
                return total_loss, separate_losses

        # Always use ordinary loss for validation
        if mode == 'valid':
            return self.ordinary_loss(p, targets)

    
    def screen_time_estimation_loss_motivate_good_detections(self, p, targets):
    
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        # Predictions with high confidence and high IoU with labels
        decent_predictions = []

         
        for i in range(self.nl):

            conf_loss = 0
            predictions = p[i]
            anchors, shape = self.anchors[i], predictions.shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            t = targets * gain

            # Define
            bc, gxy, gwh = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            (b, c) = bc.long().T  #image, class
            gij = gxy.long()

            # Append
            tboxes = torch.cat((gxy - gij, gwh), 1)
            num_targets = targets.shape[0]


            # Getting bbox x and y coordinates
            pxy = predictions[..., 0:2]
            pxy = pxy.sigmoid() * 2 - 0.5
            
            # Getting bbox width and height
            pwh = predictions[..., 2:4]
            anchors = anchors[None, :, None, None, :]
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors
            pboxes = torch.cat((pxy, pwh), -1)  # predicted box

            # Getting bbox confidence values (as probabilities)
            raw_conf = predictions[..., 4]
            conf = raw_conf.sigmoid()

            # Get bbox class probabilities values
            classes_logits = predictions[..., 5:]
            classes_probs = classes_logits.sigmoid()

            # Filter only high confidence predictions
            high_confidence_indices = (conf > self.confidence_treshold)
            target_image_decent_predictions = torch.ones_like(high_confidence_indices, device=self.device, dtype=torch.bool)


            if high_confidence_indices.sum():

                # For each target (labeled bounding box)
                for target_index in range(num_targets):
                    
                    # Get bbox coordinates, class, and image this labeled bounding box refers to
                    target_box = tboxes[target_index]
                    target_class = c[target_index]
                    target_image_index = b[target_index]

                    # Get all bboxes and confidence values from predictions
                    all_predicted_bboxes_for_image = pboxes[target_image_index]
                    high_confidence_indices_for_image = high_confidence_indices[target_image_index]

                    # Select only those predicted bboxes that have high confidence value
                    high_conf_bboxes_for_image = all_predicted_bboxes_for_image[high_confidence_indices_for_image]
                    
                    # Calculate IoU between all prediction confidences and labeled bounding box
                    ious_predictions_label = bbox_iou(high_conf_bboxes_for_image, target_box) #, CIoU=True)

                    # Find predictions with IoU bigger than treshold
                    high_iou_predictions_indices = (ious_predictions_label > self.iou_treshold)
                    if high_iou_predictions_indices.sum() == 0:
                        continue
                    
                    # We want to ignore the loss for these predictions, so we multiply it by 0s
                    potential_high_conf_objects = torch.ones_like(ious_predictions_label, device=self.device, dtype=torch.bool)
                    potential_high_conf_objects[high_iou_predictions_indices] = 0
                    target_image_decent_predictions[target_image_index, high_confidence_indices_for_image] *= potential_high_conf_objects.squeeze()


                    # -------------------------------------------------------------#
                    # Now we want to reward the network for finding good detections
                    # We calculate the reward (negative loss)

                    # Get only high IoU predictions
                    high_iou_predictions_indices = high_iou_predictions_indices.flatten()
                    high_iou_predictions = ious_predictions_label[high_iou_predictions_indices]

                    # Get confidence from all predictions in this image
                    target_image_confs = conf[target_image_index]
                    # Get confidence only from high confidence predictions
                    high_confidences = target_image_confs[high_confidence_indices_for_image]
                    # Get confidence only from predictions with high IoU with labels
                    high_iou_confidences = high_confidences[high_iou_predictions_indices]

                    # Calculate gain as negative loss
                    # Target vector is ones because we want all predictions with 
                    # with high IoU to have confidence of 1
                    ones_vector = torch.ones(high_iou_confidences.shape, device=self.device)
                    current_conf_loss = self.mse(high_iou_confidences, ones_vector)
                    current_conf_gain = 1.0 - current_conf_loss
                    conf_loss += current_conf_gain


                    # Get class probabilites from all predictions in target image 
                    target_image_class_probs = classes_probs[target_image_index]
                    # Get only probabilites from predictions with high confidence
                    high_confidence_target_image_class_probs = target_image_class_probs[high_confidence_indices_for_image]
                    # Get only probabilites from predictions with high IoU with labels
                    high_iou_target_image_class_probs = high_confidence_target_image_class_probs[high_iou_predictions_indices]
                    
                    # Make a target
                    # Correct class has target 1, incorrect classes have target 0
                    true_classes_vector = torch.full_like(high_iou_target_image_class_probs, 0, device=self.device)
                    true_classes_vector[:, target_class] = 1
                    
                    # Calculate loss and gain (1-loss)
                    current_class_loss = self.mse(high_iou_target_image_class_probs, true_classes_vector)
                    current_class_gain = 1 - current_class_loss
                    
                    # For all correct predictions we want their IoU to be as high as possible
                    # Thats why gain score is just IoU
                    lbox += (high_iou_predictions).mean()  
            
            # Scaling conf loss by layer
            total_layer_conf_loss = self.balance[i] * conf_loss
            lobj += total_layer_conf_loss


            decent_predictions.append(target_image_decent_predictions)


        #lbox *= -self.hyp['box'] / 600
        #lobj *= -self.hyp['obj'] / 10000
        #lcls *= -self.hyp['cls'] / 1000


        lbox *= -self.hyp['box'] / 60
        lobj *= -self.hyp['obj'] / 1000
        lcls *= -self.hyp['cls'] / 100

        bs = predictions.shape[0]  # batch size        
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach(), decent_predictions


    def screen_time_estimation_loss_ignore_good_detections(self, p, targets):
        
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        # Predictions with high confidence and high IoU with labels
        decent_predictions = []

        # Za svaki sloj 
        for i in range(self.nl):

            conf_loss = 0
            predictions = p[i]
            anchors, shape = self.anchors[i], predictions.shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            

            # Match targets to anchors
            t = targets * gain

            # Define
            bc, gxy, gwh = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            (b, c) = bc.long().T  #image, class
            gij = gxy.long()

            # Append
            tboxes = torch.cat((gxy - gij, gwh), 1)
            num_targets = targets.shape[0]


            # Getting bbox x and y coordinates
            pxy = predictions[..., 0:2]
            pxy = pxy.sigmoid() * 2 - 0.5
            
            # Getting bbox width and height
            pwh = predictions[..., 2:4]
            anchors = anchors[None, :, None, None, :]
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors
            pboxes = torch.cat((pxy, pwh), -1)  # predicted box

            # Getting bbox confidence values (as probabilities)
            raw_conf = predictions[..., 4]
            conf = raw_conf.sigmoid()

            # Get bbox class probabilities values
            classes_logits = predictions[..., 5:]
            classes_probs = classes_logits.sigmoid()

            # Filter only high confidence predictions
            high_confidence_indices = (conf > self.confidence_treshold)
            target_image_decent_predictions = torch.ones_like(high_confidence_indices, device=self.device, dtype=torch.bool)


            if high_confidence_indices.sum():

                # For each target (labeled bounding box)
                for target_index in range(num_targets):
                    
                    # Get bbox coordinates, class, and image this labeled bounding box refers to
                    target_box = tboxes[target_index]
                    target_class = c[target_index]
                    target_image_index = b[target_index]

                    # Get all bboxes and confidence values from predictions
                    all_predicted_bboxes_for_image = pboxes[target_image_index]
                    high_confidence_indices_for_image = high_confidence_indices[target_image_index]

                    # Select only those predicted bboxes that have high confidence value
                    high_conf_bboxes_for_image = all_predicted_bboxes_for_image[high_confidence_indices_for_image]
                    
                    # Calculate IoU between all prediction confidences and labeled bounding box
                    iou_predictions_label = bbox_iou(high_conf_bboxes_for_image, target_box) #, CIoU=True)

                    # Find predictions with IoU bigger than treshold
                    high_iou_predictions = (iou_predictions_label > self.iou_treshold)
                    if high_iou_predictions.sum() == 0:
                        continue
                    
                    # We want to ignore the loss for these predictions, so we multiply it by 0s
                    potential_high_conf_objects = torch.ones_like(iou_predictions_label, device=self.device, dtype=torch.bool)
                    potential_high_conf_objects[high_iou_predictions] = 0
                    target_image_decent_predictions[target_image_index, high_confidence_indices_for_image] *= potential_high_conf_objects.squeeze()

            decent_predictions.append(target_image_decent_predictions)
        
        return decent_predictions


    def ordinary_loss(self, p, targets, decent_predictions=None):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            if decent_predictions:
                target_layer_decent_predictions = decent_predictions[i]
                
                true_detections = (tobj > 0)
                no_detections = (tobj == 0)
                
                # Motivate true detections to have higher confidence
                positive_loss = (true_detections * self.BCEobj(pi[..., 4], tobj))

                # Motivate wrong detections to have 0 confidence
                # But ignore decent predictions (this is done by multiplying with target_layer_decent_predictions)
                zeroed_detections = torch.zeros_like(tobj, device=self.device)
                negative_loss = (no_detections * target_layer_decent_predictions * self.BCEobj(pi[..., 4], zeroed_detections))
                obji = (positive_loss + negative_loss).mean()

                
            else:
                obji = self.BCEobj(pi[..., 4], tobj).mean()

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
