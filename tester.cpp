//
// Created by husky on 11/11/2021.
//

#include <bits/stdc++.h>
using namespace std;
#include "chess.cpp"
using namespace chess;

int main(){
    Board board = Board();
    Move move1 = board.generate_legal_moves()[0];
    cout << move1 << endl;
    Move move2 = move1;
    move1 = board.generate_legal_moves()[3];
    cout << move1 << " " << move2 << endl;
}