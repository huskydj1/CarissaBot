 #include <iostream>
#include <map>
using namespace std;
#include "thc.cpp"
using namespace thc;

const float MAX_EVAL = numeric_limits<float>::max();
const float MIN_EVAL = numeric_limits<float>::min();
const int DEPTH_SEARCH = 2;
ChessRules BOARD_NULL;
Move bestMove;

void display_position(ChessRules &cr, const string &description){
    string fen = cr.ForsythPublish();
    string s = cr.ToDebugStr();
    //printf( "%s\n", description.c_str() );
    //printf( "FEN (Forsyth Edwards Notation) = %s\n", fen.c_str() );
    printf( "Position = %s\n", s.c_str() );
}

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

float evaluatePosition(ChessRules &board){
    string fen = board.ForsythPublish();
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

float dfs(ChessRules &board, int height, float alpha, float beta, bool alphaTurn){ //Return moves as well
    TERMINAL check_position_terminal;
    bool legal = board.Evaluate(check_position_terminal);
    assert(legal == true);

    if(check_position_terminal!=NOT_TERMINAL){ // Game is Over
        if(check_position_terminal == TERMINAL_WCHECKMATE){
            return MAX_EVAL;
        }
        else if(check_position_terminal == TERMINAL_BCHECKMATE){
            return MIN_EVAL;
        }
        else if(check_position_terminal == TERMINAL_WSTALEMATE
           || check_position_terminal == TERMINAL_BSTALEMATE){
            return 0;
        }
        else{
            cout << "ERROR: GAME ENDED BUT NOT CHECKMATES OR STALEMATE" << endl;
        }
    }
    else if(height == 0){ // At Leaf Node
        return evaluatePosition(board);
    }
    else{ // Game Continues
        vector<Move> moves;
        board.GenLegalMoveList(moves);

        if(alphaTurn){ // White's turn
            float maxEval = MIN_EVAL;
            Move maxMove = moves[0];

            for(Move move : moves){
                board.PlayMove(move);
                float eval = dfs(board, height - 1, alpha, beta, false);
                board.PopMove(move);

                if(maxEval < eval){
                    maxEval = eval;
                    swap(maxMove, move);
                }
                alpha = max(alpha, eval);

                if(alpha >= beta){
                    break;
                }
            }

            if(height == DEPTH_SEARCH){
                bestMove = maxMove;
            }
            return maxEval;
        }
        else{ // Black's turn
            float minEval = MAX_EVAL;
            Move minMove = moves[0];

            for(Move move : moves){
                board.PlayMove(move);
                float eval = dfs(board, height - 1, alpha, beta, true);
                board.PopMove(move);

                if(minEval > eval){
                    minEval = eval;
                    swap(minMove, move);
                }
                beta = min(beta, eval);

                if(beta <= alpha){
                    break;
                }
            }

            if(height == DEPTH_SEARCH){
                bestMove = minMove;
            }
            return minEval;
        }
    }
}

void play_OneComputerOneHuman(ChessRules &board, bool whiteFlag){
    TERMINAL check_position_terminal;
    bool legal = board.Evaluate(check_position_terminal);
    assert(legal == true);

    int i = 0;
    while (check_position_terminal == NOT_TERMINAL) {
        display_position(board, "Position");

        cout << ++i << ". " << (board.WhiteToPlay() ? "[WHITE] " : "[BLACK] ");

        if(whiteFlag==board.WhiteToPlay()){ // Bot's Turn
            float eval = dfs(board, 2, MIN_EVAL, MAX_EVAL, whiteFlag);
            cout << "BOT PLAYS: " << bestMove.NaturalOut(&board) << endl << endl;
        }
        else { // Player's Move
            string san;
            cout << "Enter Move: ";

            vector<Move> moves;
            board.GenLegalMoveList(moves);
            while (true) {
                cin >> san;
                cout << endl;
                try {
                    bestMove.NaturalIn(&board, san.c_str());
                    bool moveIsLegal = false;
                    for (Move move : moves){
                        if(bestMove == move){
                            moveIsLegal = true;
                            break;
                        }
                    }

                    if (!moveIsLegal) {
                        throw invalid_argument("");
                    }
                }
                catch (invalid_argument){
                    cout << "Invalid Move, Try Again..." << endl;
                    continue;
                }
                break;
            }
        }
        board.PlayMove(bestMove);
    }

    cout << "Game Over!" << endl;
    if(check_position_terminal == TERMINAL_WCHECKMATE){
        cout << "WHITE WINS!" << endl;
    }
    else if(check_position_terminal == TERMINAL_BCHECKMATE){
        cout << "BLACK WINS!" << endl;
    }
    else if(check_position_terminal == TERMINAL_WSTALEMATE
            || check_position_terminal == TERMINAL_BSTALEMATE){
        cout << "STALEMATE!" << endl;
    }
}


int main() {
    //display_position(BOARD_NULL, "INITIAL POSITION");
    //cout << BOARD_NULL.ForsythPublish();
    ChessRules cr;
    play_OneComputerOneHuman(cr, true);
}