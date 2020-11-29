import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model_arch(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map +=  score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class myScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, seg, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input, seg).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model_arch(input * norm_saliency_map, seg)
                output = F.softmax(output)
                score = output[0][predicted_class]

                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (27, 27)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class CAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, seg, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input, seg).cuda()
        params = list(self.model_arch.parameters())
        weight_softmax = torch.squeeze(params[-4].data)
        weight_softmax = torch.softmax(weight_softmax, dim=1)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        size_upsample = (h, w)

        activations = self.activations['value']
        b, k, u, v = activations.size()
        activations = activations.permute(0, 2, 3, 1)
        activations = activations.reshape(u*v, k)
        weight = weight_softmax[class_idx].unsqueeze(1)
        cam = torch.mm(activations, weight)
        cam = cam.reshape(u, v)
        norm_cam = (cam - cam.min()) / (cam.max() - cam.min())
        norm_cam = norm_cam.unsqueeze(0).unsqueeze(1)
        norm_cam = F.interpolate(norm_cam, size=size_upsample, mode='bilinear', align_corners=False)

        return norm_cam

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class my_CAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input,  class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input).cuda()
        params = list(self.model_arch.parameters())
        weight_softmax = torch.squeeze(params[-2].data)
        weight_softmax = torch.softmax(weight_softmax, dim=1)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        size_upsample = (h, w)

        activations = torch.relu(self.activations['value'])
        b, k, u, v = activations.size()
        activations = activations.permute(0, 2, 3, 1)
        activations = activations.reshape(u*v, k)
        weight = weight_softmax[class_idx].unsqueeze(1)
        cam = torch.mm(activations, weight)
        cam = cam.reshape(u, v)
        norm_cam = (cam - cam.min()) / (cam.max() - cam.min())
        norm_cam = norm_cam.unsqueeze(0).unsqueeze(1)
        norm_cam = F.interpolate(norm_cam, size=size_upsample, mode='bilinear', align_corners=False)

        return norm_cam

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)