#include <iostream>c
#include <map>
using namespace std;
#include "chess.cpp"
using namespace chess;

const float MAX_EVAL = numeric_limits<float>::max();
const float MIN_EVAL = numeric_limits<float>::min();
const Board BOARD_NULL = Board();
const Move MOVE_NULL = Move::null();

const map<char, int> whitePieceValue = {
        {'P', 1},
        {'R', 5},
        {'N', 3},
        {'B', 3},
        {'Q', 9},
        {'K', 0},
};
const map<char, int> blackPieceValue = {
        {'p', 1},
        {'r', 5},
        {'n', 3},
        {'b', 3},
        {'q', 9},
        {'k', 0}
};

float evaluatePosition(Board board){
    string fen = board.fen();
    int whiteMaterial = 0, blackMaterial = 0;
    for(char cur : fen){
        if(cur == ' '){
            break;
        }
        else if(cur == '/' || (1<=cur-'0' && cur-'0'<=9)){
            continue;
        }
        else if(whitePieceValue.find(cur) != whitePieceValue.end()){
            whiteMaterial += whitePieceValue.at(cur);
        }
        else if(blackPieceValue.find(cur) != blackPieceValue.end()){
            blackMaterial += blackPieceValue.at(cur);
        }
        else{
            cout << "ERROR IN EVALUATION FUNCTION" << endl;
        }
    }

    return whiteMaterial - blackMaterial;
}

pair<Move, float> dfs(Board board, int height, float alpha, float beta, bool alphaTurn, Move prevMove = MOVE_NULL){ //Return moves as well
    if(height==0 || board.is_game_over(true)){
        return make_pair(prevMove, evaluatePosition(board));
    }
    else if (alphaTurn){
        float maxEval = MIN_EVAL;
        Move maxMove = prevMove;
        for(Move move : board.generate_legal_moves()){
            board.push(move);
            float eval = dfs(board, height - 1, alpha, beta, false, move).second;
            board.pop();

            if(maxEval < eval){
                maxEval = eval;
                maxMove = move;
            }
            alpha = max(alpha, eval);
            // Alpha at this point will be >= its current value. If this is higher than a guaranteed beta, black will go for the other option in the previous turn.
            if(alpha >= beta){
                break;
            }
        }
        return make_pair(maxMove, maxEval);
    }
    else{
        float minEval = MAX_EVAL;
        Move minMove = prevMove;
        for(Move move : board.generate_legal_moves()){
            board.push(move);
            float eval = dfs(board, height - 1, alpha, beta, true, move).second;
            board.pop();

            if(minEval > eval){
                minEval = eval;
                minMove = move;
            }
            beta = min(beta, eval);
            // Beta at this point will be <= its current value. If this is lower than a guaranteed alpha, white will go for the other option in the previous move.
            if(beta <= alpha){
                break;
            }
        }
        return make_pair(minMove, minEval);
    }
}

void play_OneComputerOneHuman(Board board, bool whiteFlag){
    while (!board.is_game_over(true)) {
        while (true) {
            cout << board.unicode(false, true) << endl;

            // Move in san format
            Move move = MOVE_NULL;
            if(whiteFlag==board.turn){ //Bot's Turn
                // TODO: Decide bot's next move
                pair<Move, float> nextMove_information = dfs(board, 2, MIN_EVAL, MAX_EVAL, board.turn);
                move = nextMove_information.first;
            }
            else{
                string san;
                cout << board.ply() + 1 << ". " << (board.turn ? "[WHITE] " : "[BLACK] ") << "Enter Move: ";
                cin >> san;
                cout << endl;
                move = board.parse_san(san);
            }

            // Try Playing Move
            try {
                if (!move) {
                    throw invalid_argument("");
                }
                board.push(move);
                break;
            } catch (invalid_argument) {
                cout << "Invalid Move, Try Again..." << endl;
            }
        }
    }

    cout << "Game Over! Result: " << board.result(true);
}

void play_TwoHumans(Board board){
    while (!board.is_game_over(true)) {
        while (true) {
            cout << board.unicode(false, true) << endl;

            string san;
            cout << board.ply() + 1 << ". " << (board.turn ? "[WHITE] " : "[BLACK] ") << "Enter Move: ";
            cin >> san;
            cout << endl;

            try {
                chess::Move move = board.parse_san(san);
                if (!move) {
                    throw invalid_argument("");
                }
                board.push(move);
                break;
            } catch (invalid_argument) {
                cout << "Invalid Move, Try Again..." << endl;
            }
        }
    }

    cout << "Game Over! Result: " << board.result(true);
}

int main() {

    chess::Board board;

    //cout << board.unicode(false, true) << endl;

    //play_TwoHumans(board);
    play_OneComputerOneHuman(board, 0); //TODO: Test "play_OneComputerOneHuman" method
}