#!/usr/bin/env bash
# Call stockfish engine on mac and return only the evaluation score
# Usage stockfish.sh 'r1b2rk1/4qppp/2pp4/pp6/3Bn3/PB3Q1P/1PP2PP1/3R1RK1' 5 mac 12 1024
# Usage stockfish.sh 'r1b2rk1/4qppp/2pp4/pp6/3Bn3/PB3Q1P/1PP2PP1/3R1RK1' 5 mac 12 1024
# Assumes the stockfish binary is called 'stockfish_'+binary 

fen=$1
turn=$2
castling=$3
enpassant=$4
half_move=$5
full_move=$6
seconds=${7:-3}
binary=${8:-linux}
threads=${9:-12}
memory=${10:-1024}

(
echo "setoption name Hash value $memory" ;
echo "setoption name threads value $threads" ;
echo "position fen $fen $turn $castling $enpassant $half_move $full_move" ;
echo "go infinite";
sleep $seconds
) | stockfish > analysis.txt
cat analysis.txt | grep -ohE "score cp (-?[0-9]+)" | tail -1 | cut -d' ' -f3
cat analysis.txt | grep -ohE "score mate (-?[0-9]+)" | tail -1 | cut -d' ' -f3
rm -rf analysis.txt