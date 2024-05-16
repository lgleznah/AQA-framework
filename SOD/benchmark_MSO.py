# Benchmark methods on the MSO dataset
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
import os
import pickle
from code.get_Param import get_param
from code.SOD_class import SOD
from code.utils import do_mmr, do_nms, imread_rgb
import code.eval as eval
from code.prop_Opt import prop_opt
from tensorflow.keras import backend as K

param = get_Param.get_param(modelName = 'SOD_python', weights_path = '/content/drive/My Drive/Meero/weights/sod_cnn_weights.h5', center_path = '/content/drive/My Drive/Meero/layout_aware_subnet/SOD_python/center100.npy')
sod_model = SOD()
imgIdx = eval.load_MSO_dataset()

#cacheDir = 'propCache'
cacheDir = 'propCache'
if not os.path.isdir(cacheDir):
  os.mkdir(cacheDir)

props = []
flag = False


cacheName = 'MSO_' + param['modelName'] + '.pickle'

# load precomputed proposals if possible
if os.path.isfile(cacheDir + '/' + cacheName):
  print('using precomputed proposals')
  props = eval.read_props(cacheDir + '/' + cacheName)
  flag = True
else:
  fns = ['dataset/MSO/img/' + imgIdx[i]['name'] for i in range(len(imgIdx))]
  P_all, S_all = sod_model.predict_for_benchmarks(fns)


legs = []

## evaluate the MAP method
print('run the MAP method')
lambda_ = [0, 0.000001, 0.0001]
lambda_.extend(list(np.arange(0.01, 0.1, 0.01)))
lambda_.extend(list(np.arange(0.1, 1, 0.1)))
res = [[[] for j in range(len(imgIdx))] for i in range(len(lambda_))]
for j in range(len(imgIdx)):
  I = imread_rgb(['dataset/MSO/img/' + imgIdx[j]['name']])[0]
  imsz = np.array([I.shape[0], I.shape[1]])

  # load precomputed proposals
  if flag:
    P = props[j]['P']
    S = props[j]['S']
  else:
    props.append({})
    P = P_all[j]
    S = S_all[j]
    props[j]['P'] = P
    props[j]['S'] = S

  if j % 100 == 0:
    print('processed {} images'.format(j))


  for i in range(len(lambda_)):
    param['lambda'] = lambda_[i]
    param['gamma'] = 10 * lambda_[i]
    tmpRes, _ = prop_Opt.prop_opt(P, S, param)

    # scale bboxes to full size
    tmpRes = tmpRes * np.tile(np.roll(imsz, 1), 2).reshape(-1, 1)
    res[i][j] = tmpRes


K.clear_session()

if not flag:
  eval.save_props(props, cacheDir + '/' + cacheName)

plt.figure()
TP, NPred, NGT = eval.evaluate_bbox(imgIdx, res.copy())

P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(), 0.01)
plt.plot(R, P, 'r')
ap = eval.calc_ap(R, P)
legs.append('MAP: {}'.format(ap))

## evaluate the NMS baseline
print('run the NMS baseline')
thresh = np.arange(0, 1, 0.02)
res = [[[] for j in range(len(imgIdx))] for i in range(len(thresh))]
for j in range(len(imgIdx)):
  I = imread_rgb(['dataset/MSO/img/' + imgIdx[j]['name']])[0]
  imsz = np.array([I.shape[0], I.shape[1]])
  P = props[j]['P']
  S = props[j]['S']
  if j % 100 == 0:
    print('processed {} images'.format(j))

  # scale bboxes to full size
  P = P * np.tile(np.roll(imsz, 1), 2).reshape(-1, 1)
  idx = np.argsort(-S)
  S = np.take(S, idx)
  P = np.take(P, idx, axis = 1)
  P, sidx = do_nms(P, 0.4)
  S = S[sidx]

  for i in range(len(thresh)):
    tmpRes = P[:, S >= thresh[i]]
    res[i][j] = tmpRes

TP, NPred, NGT = eval.evaluate_bbox(imgIdx, res)
P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(), 0.01)
plt.plot(R, P, 'b')
ap = eval.calc_ap(R,P)
legs.append('NMS: {}'.format(ap))

## evaluate the MMR baseline
print('run the MMR  baseline')
thresh = np.arange(-1.0, 1.0, 0.01)
res = [[[] for j in range(len(imgIdx))] for i in range(len(thresh))]
for j in range(len(imgIdx)):
  I = imread_rgb(['dataset/MSO/img/' + imgIdx[j]['name']])[0]
  imsz = np.array([I.shape[0], I.shape[1]])
  P = props[j]['P']
  S = props[j]['S']
  if j % 100 == 0:
    print('processed {} images'.format(j))


  # scale bboxes to full size
  P = P * np.tile(np.roll(imsz, 1), 2).reshape(-1, 1)
  idx = np.argsort(-S)
  S = np.take(S, idx)
  P = np.take(P, idx, axis = 1)
  [P, S] = do_mmr(P.T, S, 1.0)
  for i in range(len(thresh)):
    tmpRes = P[:, S > thresh[i]]
    res[i][j] = tmpRes

TP, NPred, NGT = eval.evaluate_bbox(imgIdx, res)
P = TP.sum(1) / np.maximum(NPred.sum(1), 0.01)
R = TP.sum(1) / np.maximum(NGT.sum(), 0.01)
plt.plot(R,P,'g')
ap = eval.calc_ap(R,P)
legs.append('MMR: {}'.format(ap))

plt.grid()
plt.legend(legs)
plt.title('PR Curves on the MSO Dataset ({})'.format(param['modelName']))
