import torch
from model import CarissaNet
import data_manip
from data_manip import convert_to_bitboard

model = CarissaNet(blocks=8, filters=64, se_channels=32)

sdict = torch.load('models/model_111721_newdata_8x64_35.pt', map_location='cpu')
for key in list(sdict.keys()):
    if key.startswith("module."):
        sdict[key[7:]] = sdict.pop(key)

model.load_state_dict(sdict)

if torch.cuda.is_available():
    model = model.cuda()

example = torch.randn(100, 29, 8, 8)
fen = 'rnbqk1Q1/pppp1p1p/5bp1/8/4P3/8/PPPP1PPP/RNB1KBNR b KQq - 0 5'
model_input = torch.unsqueeze(torch.Tensor(convert_to_bitboard(fen)), 0)

traced_script_module = torch.jit.trace(model, example)
# print(traced_script_module(model_input))

traced_script_module.save('models/traced_model_111721_newdata_8x64_35.pt')