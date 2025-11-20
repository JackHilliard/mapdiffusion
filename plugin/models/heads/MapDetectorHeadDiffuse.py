import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss

from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from ..utils.memory_buffer import StreamTensorMemory
from ..utils.coef import forward_steps


def gt_padding(gts):
    all_labels_list = []
    all_lines_list = []
    all_scores_list = []
    bs = len(gts['labels'])
    for i in range(bs):
        gts_scores = []
        gts_class = gts['labels'][i]
        gts_poly = gts['lines'][i]
        num_gts = len(gts_class)
        for k in range(num_gts):
            tem_scores = np.zeros(3)+0.0
            tem_scores[gts_class[k]] = 1.0
            gts_scores.append(tem_scores)
        for j in range(100-num_gts):
            random_class = np.random.randint(0, 3)
            tem_scores = np.zeros(3)+0.0
            tem_scores[random_class] = 0.0
            gts_scores.append(tem_scores)
            # Perform Gaussian padding
            out_poly = np.clip(np.random.normal(loc=0.5, scale=0.25, size=(20,2)), 0, 1)
            gts_class.append(random_class)
            gts_poly.append(out_poly)
        indices = np.arange(100)
        np.random.shuffle(indices)
        gts_poly = np.array(gts_poly,dtype=np.float32)[indices]
        gts_scores = np.array(gts_scores,dtype=np.float32)[indices]
        gts_class = np.array(gts_class)[indices]
        all_labels_list.append(gts_class)
        all_lines_list.append(gts_poly)
        all_scores_list.append(gts_scores)
    return all_lines_list, all_labels_list, all_scores_list


def timesteps_to_tensor(ts: int or list[int], batch_size):
    if isinstance(ts, list):
        assert batch_size % len(ts) == 0, "batch_size must be divisible by length of timesteps list"
    
    if isinstance(ts, int):
        return ts * torch.ones(batch_size)
    else:
        mini_batch_size = batch_size // len(ts)
        return torch.cat([ts[i] * torch.ones(mini_batch_size) for i in range(len(ts))])


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@HEADS.register_module(force=True)
class MapDetectorHeadDiffuse(nn.Module):

    def __init__(self, 
                 num_queries,
                 num_classes=3,
                 in_channels=128,
                 embed_dims=256,
                 score_thr=0.1,
                 num_points=20,
                 coord_dim=2,
                 roi_size=(60, 30),
                 different_heads=True,
                 predict_refine=False,
                 bev_pos=None,
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 streaming_cfg=dict(),
                 transformer=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict()
                ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.bev_pos = bev_pos
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight

        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.transformer = build_transformer(transformer)

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.assigner = build_assigner(assigner)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self._init_embedding()
        self._init_branch()
        self.init_weights()


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        self.transformer.init_weights()

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            if isinstance(self.cls_branches, nn.ModuleList):
                for m in self.cls_branches:
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, bias_init)
            else:
                m = self.cls_branches
                nn.init.constant_(m.bias, bias_init)
        

    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims//2,
            normalize=True
        )
        self.bev_pos_embed = build_positional_encoding(positional_encoding)
        
        # diffusion_embedding
        self.input_dim = 128
        time_embed_dim = self.input_dim * 4
        self.coords_embed = nn.Linear(2*self.input_dim, self.embed_dims)
        self.time_embed = nn.Sequential(
            nn.Linear(self.input_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.reference_points = nn.Linear(self.embed_dims+time_embed_dim, self.num_points*2)
        
        # query_pos_embed & query_embed
        self.geometry_branch = nn.Sequential(
            Linear(self.num_points * self.coord_dim, 2*self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(2*self.embed_dims),
            Linear(2*self.embed_dims, 2*self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(2*self.embed_dims),
            Linear(2*self.embed_dims, self.embed_dims),
            nn.ReLU(),
        )
        
    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [
            Linear(self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, self.num_points * self.coord_dim),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        num_layers = self.transformer.decoder.num_layers
        if self.different_heads:
            cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings # (bs, embed_dims, H, W)
    
        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    def forward_train(self,coef, total_steps, inputs, bev_features, img_metas, gts):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        pos_embed = None

        device = torch.device('cuda')
        polylines, poly_class, poly_scores = gt_padding(inputs)
        polylines = torch.tensor(polylines, dtype = torch.float32).to(device)  # [B, num_q, num_points, num_coords]
        t = np.random.randint(1, total_steps + 1)
        # Generate random vectorized polyline coordinates between 0 and 1
        noise = torch.normal(mean=0.5, std=0.25, size=polylines.shape).to(device)
        noise = noise.clip(0, 1)
        query_coords = forward_steps(polylines,t,noise,coef)  # [B, num_q, num_points, num_coords]
        vert_coords = rearrange(query_coords, 'b n1 n2 c -> b n1 (n2 c)') # [B, num_q, 40]
        vert_encodings = self.geometry_branch(vert_coords)  # encode coordinates to [B, num_q, 512]
        timestep = timesteps_to_tensor(t,bs)
        t_emb = self.time_embed(timestep_embedding(timestep.to(device), self.input_dim)).to(device)#[b,512]
        repeat_t_emb = repeat(t_emb, 'b n2 -> b n1 n2', n1=vert_encodings.shape[1])#[b,num_q,512]
        reference_points = vert_coords.clip(0, 1)

        rf_encodings = torch.cat([vert_encodings, repeat_t_emb], dim=-1) #[b,num_q,1024]

        reference_points_offset = self.reference_points(rf_encodings).sigmoid() #[b,num_q,1024]
        reference_points_offset = reference_points_offset * 2 - 1
        reference_points = (reference_points + reference_points_offset).clip(0, 1)
        init_reference_points = reference_points.view(bs, self.num_queries, self.num_points, 2)
        
        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
        assert list(vert_encodings.shape) == [bs, self.num_queries, self.embed_dims]
        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            t_emb=t_emb,
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=vert_encodings,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            query_key_padding_mask=vert_encodings.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn,
        )
        outputs = []
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list
            }
            outputs.append(pred_dict)
        
        loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list = self.loss(gts=gts, preds=outputs)

        return outputs, loss_dict, det_match_idxs, det_match_gt_idxs
    
    def forward_test(self, query_coords, timestep, bev_features, img_metas):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''
        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        pos_embed = None

        device = torch.device('cuda')
        vert_coords = rearrange(query_coords, 'b n1 n2 c -> b n1 (n2 c)') # [B, num_q, 40]
        vert_encodings = self.geometry_branch(vert_coords)  # [B, num_q, 512]
        timestep = timesteps_to_tensor(timestep,bs)
        t_emb = self.time_embed(timestep_embedding(timestep.to(device), self.input_dim)).to(device) # [b, 512]
        repeat_t_emb = repeat(t_emb, 'b n2 -> b n1 n2', n1=vert_encodings.shape[1]) # [b, num_q, 512]
        reference_points = vert_coords.clip(0, 1)
        rf_encodings = torch.cat([vert_encodings, repeat_t_emb], dim=-1) #[b, num_q, 1024]

        reference_points_offset = self.reference_points(rf_encodings).sigmoid() # [b, num_q, 1024]
        reference_points_offset = reference_points_offset * 2 - 1
        reference_points = (reference_points + reference_points_offset).clip(0, 1)
        init_reference_points = reference_points.view(bs, self.num_queries, self.num_points, 2)
       
        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
        assert list(vert_encodings.shape) == [bs, self.num_queries, self.embed_dims]

        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            t_emb = t_emb,
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=vert_encodings,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            query_key_padding_mask=vert_encodings.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn
        )

        outputs = []
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            prop_mask_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])
                prop_mask = scores.new_ones((len(scores[i]), ), dtype=torch.bool)
                prop_mask[-self.num_queries:] = False
                prop_mask_list.append(prop_mask)

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'prop_mask': prop_mask_list
            }
            outputs.append(pred_dict)

        return outputs

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].
                
            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result, gt_permute_idx = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
                (num_pred_lines, ), self.num_classes, dtype=torch.long) # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines) # (num_q, )

        lines_target = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        
        if num_gt > 0:
            if gt_permute_idx is not None: # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype) # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype) # (num_q, 2*num_pts)
        
        lines_weights[pos_inds] = 1.0 # (num_q, 2*num_pts)

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']

        lines_pred = preds['lines']

        (labels_list, label_weights_list,
        lines_targets_list, lines_weights_list,
        pos_inds_list, neg_inds_list,pos_gt_inds_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, gt_bboxes_ignore=gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list, # list[Tensor(num_q, )], length=bs
            label_weights=label_weights_list, # list[Tensor(num_q, )], length=bs, all ones
            lines=lines_targets_list, # list[Tensor(num_q, 2*num_pts)], length=bs
            lines_weights=lines_weights_list, # list[Tensor(num_q, 2*num_pts)], length=bs
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds,
                    gts,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list =\
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.cls_out_channels) # (bs*num_q, cls_out_channels)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1) # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1) # (bs*num_q, )
        
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)
        
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)
        gt_lines = torch.cat(new_gts['lines'], dim=0)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list, new_gts['lines']
    
    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             reduction='mean'):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }
                    
                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list = multi_apply(
            self.loss_single, preds, gts, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_lines_list
    
    def post_process(self, preds_dict, tokens, thr=0.0):
        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores'] # (bs, num_queries, 3)
        prop_mask = preds_dict['prop_mask']

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            tmp_prop_mask = prop_mask[i]
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)
            # focal loss
            if self.loss_cls.use_sigmoid:
                tmp_scores, tmp_labels = scores[i].max(-1)
                tmp_scores = tmp_scores.sigmoid()
                pos = tmp_scores > thr
            else:
                assert self.num_classes + 1 == self.cls_out_channels
                tmp_scores, tmp_labels = scores[i].max(-1)
                bg_cls = self.cls_out_channels
                pos = tmp_labels != bg_cls

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_mask = tmp_prop_mask[pos]

            if len(tmp_scores) == 0:
                single_result = {
                'vectors': [],
                'scores': [],
                'labels': [],
                'prop_mask': [],
                'token': tokens[i]
            }
            else:
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'prop_mask': tmp_prop_mask.detach().cpu().numpy(),
                    'token': tokens[i]
                }
            results.append(single_result)
        
        return results

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)