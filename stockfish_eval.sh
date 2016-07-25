#!/usr/bin/env bash
# Call stockfish engine on mac and return only the evaluation score
# Usage stockfish.sh 'r1b2rk1/4qppp/2pp4/pp6/3Bn3/PB3Q1P/1PP2PP1/3R1RK1' 5 mac 12 1024
# Usage stockfish.sh 'r1b2rk1/4qppp/2pp4/pp6/3Bn3/PB3Q1P/1PP2PP1/3R1RK1' 5 mac 12 1024
# Assumes the stockfish binary is called 'stockfish_'+binary 

fen=$1
seconds=${2:-3}
binary=${3:-linux}
threads=${4:-12}
memory=${5:-1024}

(
echo "setoption name Hash value $memory" ;
echo "setoption name threads value $threads" ;
echo "position fen $fen" ;
echo "go infinite";
sleep $seconds
) | stockfish > analysis.txt
cat analysis.txt | grep -ohE "score cp (-?[0-9]+)" | tail -1 | cut -d' ' -f3
cat analysis.txt | grep -ohE "score mate (-?[0-9]+)" | tail -1 | cut -d' ' -f3
rm -rf analysis.txt