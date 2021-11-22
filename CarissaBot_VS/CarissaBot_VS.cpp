#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
using namespace std;

#include "thc.cpp"
using namespace thc;

torch::Tensor convert_to_bitboard(ChessRules &board) {
    // Convert https://github.com/huskydj1/CarissaBot/blob/0c7d79cf83684c5376515f319ea43677ee00737f/model/data_manip.py#L49 to c++
    // Example of usage: https://github.com/huskydj1/CarissaBot/blob/main/model/torch_script_model.py
    torch::Tensor bitboard = torch::zeros((29, 8, 8));
    
    bool turn_color = board.WhiteToPlay();
    bitboard[0, :, : ] = turn_color;

    
}

int main() {
    // Resources for libtorch: - https://github.com/alantess/learntorch/blob/main/torchscript/L1/main.cpp
    //                         - https://pytorch.org/tutorials/advanced/cpp_export.html#step-4-executing-the-script-module-in-c


    std::string traced_model = "D:/Documents/202122_Andover/Fall/CSC630/CarissaBot/traced_model_111721_newdata_8x64_35.pt";

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(traced_model);

        //string fen = "rnbqk1Q1/pppp1p1p/5bp1/8/4P3/8/PPPP1PPP/RNB1KBNR b KQq - 0 5";
        ChessRules board;
        torch::Tensor model_input = torch::unsqueeze(convert_to_bitboard(board), 0);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}