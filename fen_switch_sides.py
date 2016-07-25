fen = '4r1k1/ppp1nr2/3q2p1/3P4/3NK1P1/1Q1P3P/PP1B2B1/RN5R'
new_fen = ''

for char in fen:
	if char.isupper():
		new_fen += char.lower()
	elif char.islower():
		new_fen += char.upper()
	else:
		new_fen += char

print new_fen