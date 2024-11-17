def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):

oup = m

if isinstance(m, nn.Conv2d):

m: nn.Conv2d

bias = m.bias is not None

oup = SparseConv2d(

m.in_channels, m.out_channels,

kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,

dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,

)

oup.weight.data.copy_(m.weight.data)

if bias:

oup.bias.data.copy_(m.bias.data)

elif isinstance(m, nn.MaxPool2d):

m: nn.MaxPool2d

oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)

elif isinstance(m, nn.AvgPool2d):

m: nn.AvgPool2d

oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)

elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):

m: nn.BatchNorm2d

oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)

oup.weight.data.copy_(m.weight.data)

oup.bias.data.copy_(m.bias.data)

oup.running_mean.data.copy_(m.running_mean.data)

oup.running_var.data.copy_(m.running_var.data)

oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)

if hasattr(m, "qconfig"):

oup.qconfig = m.qconfig

elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):

m: nn.LayerNorm

oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)

oup.weight.data.copy_(m.weight.data)

oup.bias.data.copy_(m.bias.data)

elif isinstance(m, (nn.Conv1d,)):

raise NotImplementedError

for name, child in m.named_children():

oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))

del m

return oup