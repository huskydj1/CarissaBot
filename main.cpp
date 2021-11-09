#include "chess.cpp"
#include <iostream>
using namespace std;

int main() {
    chess::Board board;
    cout << board.unicode(false, true) << endl;

    board = chess::Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    cout << board.unicode(false, true) << endl;

    /*
    while (!board.is_game_over(true)) {
        while (true) {
            std::cout << board.unicode(false, true) << std::endl;

            std::string san;
            std::cout << board.ply() + 1 << ". " << (board.turn ? "[WHITE] " : "[BLACK] ") << "Enter Move: ";
            std::cin >> san;
            std::cout << std::endl;

            try {
                chess::Move move = board.parse_san(san);
                if (!move) {
                    throw std::invalid_argument("");
                }
                board.push(move);
                break;
            } catch (std::invalid_argument) {
                std::cout << "Invalid Move, Try Again..." << std::endl;
            }
        }
    }

    std::cout << "Game Over! Result: " << board.result(true);
     */
}