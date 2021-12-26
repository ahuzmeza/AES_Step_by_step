#! /usr/local/bin/python3
import numpy as np
from copy import deepcopy

from numpy.lib.utils import who

Sbox = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
		0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
		0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
		0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
		0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
		0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
		0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
		0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
		0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
		0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
		0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
		0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
		0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
		0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
		0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
		0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]

SboxInv = [
		0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
		0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
		0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
		0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
		0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
		0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
		0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
		0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
		0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
		0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
		0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
		0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
		0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
		0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
		0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
		0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

Rcon = [0x00000000, 0x01000000, 0x02000000,
		0x04000000, 0x08000000, 0x10000000,
		0x20000000, 0x40000000, 0x80000000,
		0x1b000000, 0x36000000]

def print_matrix(matrix):
	a = np.array(matrix)
	for line in a:
		print('  '.join(map(str, line)))

#used to display the keywords neatly in this form: w0 = 0f 15 71 c9
def print_keywords(w):
	print("Keywords:\n--------------------------------------------------------------------")
	for i in range(len(w)):
		print(f"w[{str(i)}]:\t", end='')
		for j in range(4):
			print(w[i][j], end='')
		print()

def print_round_keys(rounkeys):
	print("\nRounded keys:\n--------------------------------------------------------------------")
	# RoundeKeys = split w into 10 sublists of 4 words
	for i in range(len(rounkeys)):
		print(f"Round_{i}:\t {rounkeys[i]}")
	print()

def print_rk_state_matrix(rounkeys):
	print("Round Keys State Matricies:\n--------------------------------------------------------------------")
	for i in range(len(rounkeys)):
		a = rounkeys[i]
		n = 2
		print(f"RK_{i}: {rounkeys[i]}", end='')
		x = [a[i:i+n] for i in range(0, len(a), n)]	
		my_matrix = np.array(x).reshape(4, 4).T	

		print("\nRK_state = ")
		for i in range(4):
			print("\t", end='')
			for j in range(4):
				print(my_matrix[i][j] + ' ', end=' ')
			print()
		print()

def print_ciphertext(state_matrix):
	print("=======> CIPHERTEXT: ", end='')
	for i in range(4):
		for j in range(4):
			print(f"{state_matrix[j][i]}", end=' ')
	print()

def print_matrix_as_list(state_matrix):
	for i in range(4):
		for j in range(4):
			print(f"{state_matrix[j][i]}", end=' ')
	print()


def int_from_hex(hexdigits):
    return int(hexdigits, 16)


def mul_by_02(num):
	if num < 0x80:
		res =  (num << 1)
	else:
		res =  (num << 1)^0x1b
	return res % 0x100

def mul_by_03(num):
    return mul_by_02(num)^num

def mul_by_09(num):
    # return mul_by_03(num)^mul_by_03(num)^mul_by_03(num) - works wrong, I don't know why
    return mul_by_02(mul_by_02(mul_by_02(num))) ^ num


def mul_by_0b(num):
    # return mul_by_09(num)^mul_by_02(num)
    return mul_by_02(mul_by_02(mul_by_02(num))) ^ mul_by_02(num) ^ num


def mul_by_0d(num):
    # return mul_by_0b(num)^mul_by_02(num)
    return mul_by_02(mul_by_02(mul_by_02(num))) ^ mul_by_02(mul_by_02(num)) ^ num


def mul_by_0e(num):
    # return mul_by_0d(num)^num
    return mul_by_02(mul_by_02(mul_by_02(num))) ^ mul_by_02(mul_by_02(num)) ^ mul_by_02(num)


# rows are shifted cyclically to the left by offsets of 0,1,2, and 3
def shift_rows(state_matrix, inv):
	if inv is False:
		state_matrix[1] = np.roll(state_matrix[1], -1)
		state_matrix[2] = np.roll(state_matrix[2], -2)
		state_matrix[3] = np.roll(state_matrix[3], -3)
	else:
		state_matrix[1] = np.roll(state_matrix[1], +1)
		state_matrix[2] = np.roll(state_matrix[2], +2)
		state_matrix[3] = np.roll(state_matrix[3], +3)
	return state_matrix

def matrix_xor_matrix(state_matrix, rk_matrix):
	# xor state matrix with round key matrix
	rez_matrix = deepcopy(state_matrix)
	for i in range(4):
		for j in range(4):
			rez_matrix[i][j] = str_xor_str(state_matrix[i][j], rk_matrix[i][j])
	return rez_matrix

# takes two hex values and calculates hex1 xor hex2
def hex_xor_hex(hex1, hex2):
	# convert to binary
	bin1 = hex_to_binary(hex1)
	bin2 = hex_to_binary(hex2)

	#calculate
	xord = int(bin1, 2) ^ int(bin2, 2)

	#cut prefix
	hexed = hex(xord)[2:]

	#leading 0s get cut above, if not length 8 add a leading 0
	if len(hexed) != 8:
		hexed = '0'*(8-len(hexed)) + hexed

	return hexed

def str_xor_str(a, b):    # xor two hex strings of the same length
    return "".join(["%x" % (int(x,16) ^ int(y,16)) for (x, y) in zip(a, b)])

#takes a hex value and returns binary
def hex_to_binary(hex):
	return bin(int(str(hex), 16))


#takes from 1 to the end, adds on from the start to 1
def rot_word(word):
	return word[1:] + word[:1]


#selects correct value from sbox based on the current word
def sub_4_words_sbox(word):
	_4_w = word
	sWord = ()
	for i in range(4):
		w = _4_w[i]
		if len(w) == 1:
			w = '0'+w
		# check first char, if its a letter(a-f) get corresponding decimal
		# otherwise just take the value and add 1
		if w[0].isdigit() == False:
			row = ord(w[0]) - 86
		else:
			row = int(w[0])+1
		# repeat above for the seoncd char
		if w[1].isdigit() == False:
			col = ord(w[1]) - 86
		else:
			col = int(w[1])+1
		# get the index base on row and col (16x16 grid)
		sBoxIndex = (row*16) - (17-col)
		# get the value from sbox without prefix
		piece = hex(Sbox[sBoxIndex])[2:]
		# check length to ensure leading 0s are not forgotton
		if len(piece) != 2:
			piece = '0' + piece
		# form tuple
		sWord = (*sWord, piece)
	#return string
	return ''.join(sWord)

def sub_one_word_sbox(word, inv):
	if len(word) == 1:
		# add 0 in front of word
		word = '0' + word
	sWord = ()
	if word[0].isdigit() == False:
		row = ord(word[0]) - 86
	else:
		row = int(word[0])+1
	# repeat above for the seoncd char
	if word[1].isdigit() == False:
		col = ord(word[1]) - 86
	else:
		col = int(word[1])+1
 	# get the index base on row and col (16x16 grid)
	sBoxIndex = (row*16) - (17-col)
	# get the value from sbox without prefix
	piece = hex(Sbox[sBoxIndex])[2:]
	if inv is True:
		piece = hex(SboxInv[sBoxIndex])[2:]
	# check length to ensure leading 0s are not forgotton
	if len(piece) != 2:
		piece = '0' + piece
  	# form tuple
	sWord = (*sWord, piece)
	# return string
	return ''.join(sWord)

def matrix_sub_sbox(state_matrix, inv):
	for i in range(4):
		for j in range(4):
			state_matrix[i][j] = sub_one_word_sbox(state_matrix[i][j], inv)
	return state_matrix


def create_rk_state_matrix_from_rk(roundkey):
	# create matrix out of roundkey string
	n = 2
	x = [roundkey[i:i+n] for i in range(0, len(roundkey), n)]
	return np.array(x).reshape(4, 4).T


# define Mix Columns function for AES encryption
def mix_columns(state_matrix, inv):
	if inv is False:
		fixed_matrix = [[0x02, 0x03, 0x01, 0x01],
						[0x01, 0x02, 0x03, 0x01],
						[0x01, 0x01, 0x02, 0x03],
						[0x03, 0x01, 0x01, 0x02]]
		fixed_matrix = np.array(fixed_matrix, dtype=np.str_)
		rez_matrix = [[0x00, 0x00, 0x00, 0x00],
						[0x00, 0x00, 0x00, 0x00],
						[0x00, 0x00, 0x00, 0x00],
						[0x00, 0x00, 0x00, 0x00]]
		for i in range(4):
			for j in range(4):
				rez_l = []
				for k in range(4):
					a = fixed_matrix[i][k]
					b = state_matrix[k][j]
					rez = b
					print(f"({i},{k})*({k},{j}) ", end='')
					if a == '1':
						rez = int_from_hex(b)
						print(f"=> ({k}) = 1 * {np.binary_repr(int_from_hex(b))} = {np.binary_repr(rez)}")
					elif a == '2':
						b = (int_from_hex(b))
						rez = mul_by_02(b)
						print(f"=> ({k}) = 2*{b} = 10 * {np.binary_repr(b)} = {np.binary_repr(rez)}")
					elif a == '3':
						b = (int_from_hex(b))
						rez = mul_by_03(b)
						print(f"=> ({k}) = (2*{b})⊕{b} = {np.binary_repr(0x02 * b)} ⊕ {np.binary_repr(b)} = {np.binary_repr((rez))}")
					rez_l.append(rez)
				final_rez = rez_l[0] ^ rez_l[1] ^ rez_l[2] ^ rez_l[3]
				print(f"\nstate_matrix[{i}][{j}] = (0)⊕(1)⊕(2)⊕(3)\n  {np.binary_repr(rez_l[0], width=8)}⊕\n  {np.binary_repr(rez_l[1], width=8)}⊕\n  {np.binary_repr(rez_l[2], width=8)}⊕\n  {np.binary_repr(rez_l[3], width=8)}\n= {np.binary_repr(final_rez, width=8)} | {final_rez} | {'{:02x}'.format(final_rez)}\n")
				final_rez = '{:02x}'.format(final_rez)
				rez_matrix[i][j] = final_rez 
		state_matrix = deepcopy(rez_matrix)
	else:
		# state represents state_matrix as 'int' from hex
		state = [ [0 for i in range(4)] for i in range(4)]
		for i in range(4):
			for j in range(4):
				state[i][j] = int_from_hex(state_matrix[i][j])
		for i in range(4):													
			s0 = mul_by_0e(state[0][i]) ^ mul_by_0b(state[1][i]) ^ mul_by_0d(state[2][i]) ^ mul_by_09(state[3][i])
			s1 = mul_by_09(state[0][i]) ^ mul_by_0e(state[1][i]) ^ mul_by_0b(state[2][i]) ^ mul_by_0d(state[3][i])
			s2 = mul_by_0d(state[0][i]) ^ mul_by_09(state[1][i]) ^ mul_by_0e(state[2][i]) ^ mul_by_0b(state[3][i])
			s3 = mul_by_0b(state[0][i]) ^ mul_by_0d(state[1][i]) ^ mul_by_09(state[2][i]) ^ mul_by_0e(state[3][i])
			state_matrix[0][i] = '{:02x}'.format(s0)
			state_matrix[1][i] = '{:02x}'.format(s1)
			state_matrix[2][i] = '{:02x}'.format(s2)
			state_matrix[3][i] = '{:02x}'.format(s3)
	return state_matrix


def key_expansion_schedule(key):
	print("\n• Round Constant:")
	for i in range(1,11):
		print(f" rcon_{i}: {hex(Rcon[i])[:4]}")
	words = [()]*44
	# fill out first 4 words based on the key
	for i in range(4):
		words[i] = (key[4*i], key[4*i+1], key[4*i+2], key[4*i+3])
	
	print("\nKey Expansion:\n--------------------------------------------------------------------")
	print("• first 4 words ar first 4 sublets of 8 chars in key")
	print("• w[i] = if i mod 4 \n\t = 0, w[i-4] ⊕ g(w[-1]) \n\t!= 0, w[i-2] ⊕ w[i-1]\n")
	for i in range(4):
		print(f"w[{i}]: {''.join(words[i])}")
	# fill out the rest based on previews words, rotword, subword and rcon values
	for i in range(4, 44):
		#get required previous keywords
		temp = words[i-1]
		word = words[i-4]
		temp_index = i-1
		word_index = i-4
		# if multiple of 4 use rot, sub, rcon etc
		if i % 4 == 0:
			x = rot_word(temp)
			y = sub_4_words_sbox(x)
			rcon = Rcon[int(i/4)]
			#############
			#print(f"w[{temp_index}] = ",''.join(temp))  
			print(f"\ng(w[{temp_index}]) = ?")
			print(f"\t• w[{temp_index}] as 'w'")
			print(f"\t• rcon_{int(i/4)} = {hex(rcon)[:4]}")
			print(f"\t1) <-LShift w: {''.join(x)}")
			print(f"\t2) SubSbox w: {''.join(y)}")
			print(f"\t3) w * rcon_{int(i/4)}")
			print(f"\t{''.join(y)} ⊕ {hex(rcon)[:4]} = {''.join(temp)}")
			print(f"\tg(w[{temp_index}])] = ", ''.join(temp))  
			temp = hex_xor_hex(y, hex(rcon)[2:])
			print(f"w[{temp_index+1}] = w[{word_index}] ⊕ g(w[{temp_index}]) =",''.join(temp))
			print()
		else:
			print(f"w[{i}] = w[{i-2}] ⊕ w[{i-1}] =",''.join(temp))  
		# creating strings of hex rather than tuple
		word = ''.join(word)
		temp = ''.join(temp)
		# xor the two hex values		print(f"temp: {temp} type: {type(temp)}") 
		xord = str_xor_str(word, temp)
		words[i] = (xord[:2], xord[2:4], xord[4:6], xord[6:8])
	print()
	return words


def add_rounkeys(roundkeys, state_matrix, inv):
	print("Add Round Keys:\n--------------------------------------------------------------------")
	# create matrix out of first round key
	rk_matrix = create_rk_state_matrix_from_rk(roundkeys[0])
	who_state = "plaintext as matrix"
	rk_index = 0
	if inv is True: 
		rk_index = 10
		who_state = "encrypted text as matrix"
	print("Add Roundkey Round 0 ===============================================")
	print(" state matrix: (%s)" % who_state)
	print_matrix(state_matrix)
	print(f" ⊕ RK_matrix[{rk_index}]:")
	print_matrix(rk_matrix)
	print("=======>state_matrix<=======")
	state_matrix = matrix_xor_matrix(state_matrix, rk_matrix)
	print_matrix(state_matrix)	
 	
	if inv is False:
		for i in range(1,10):
			print(f"\nAdd Roundkey Round {i} ===============================================")
			print(" state_matrix:")
			print_matrix(state_matrix)
			print(" 1) Substitution S-box =>")
			state_matrix = matrix_sub_sbox(state_matrix, inv)
			print_matrix(state_matrix)
			print(" 2) Rows are shifted <left by offsets of 0,1,2, and 3 =>")
			state_matrix = shift_rows(state_matrix, inv)
			print_matrix(state_matrix)
			print("\n 3) Mix Column:\nmultiplies 'fixed_matrix' ⊕ state_matrix\n i_j   i_j")
			state_matrix = mix_columns(state_matrix, inv)
			print("=======>state_matrix<=======")
			print_matrix(state_matrix)
			print(f" 4) ⊕ RK_matrix[{rk_index}]:")
			rk_matrix = create_rk_state_matrix_from_rk(roundkeys[i])	
			print_matrix(rk_matrix)
			state_matrix = matrix_xor_matrix(state_matrix, rk_matrix)
			print("=======>state_matrix<=======")
			print_matrix(state_matrix)
			print_ciphertext(state_matrix)
	
		print(f"\nAdd Roundkey Round 10 ===============================================")	
		print(" 1) Substitution S-box =>")
		state_matrix = matrix_sub_sbox(state_matrix, inv)
		print_matrix(state_matrix)
		print(" 2) Rows are shifted <-left by offsets of 0,1,2, and 3 =>")
		state_matrix = shift_rows(state_matrix, inv)
		print_matrix(state_matrix)
		print(f" 3) ⊕ RK_matrix[10]:")
		rk_matrix = create_rk_state_matrix_from_rk(roundkeys[10])
		print_matrix(rk_matrix)
		state_matrix = matrix_xor_matrix(state_matrix, rk_matrix)
		print("=======>state_matrix<=======")
		print_matrix(state_matrix)
		print_ciphertext(state_matrix)
		print()
	else:		
		print(" 2) Rows are shifted right-> by offsets of 0,1,2, and 3 =>")
		state_matrix = shift_rows(state_matrix, inv)
		print_matrix(state_matrix)
		print(" 3) Substitution S-box =>")
		state_matrix = matrix_sub_sbox(state_matrix, inv)
		print_matrix(state_matrix)
		for i in range(1,10):
			print(f"\nAdd Roundkey Round {i} ===============================================")
			print(" state_matrix:")
			print_matrix(state_matrix)
			print(f" 1) ⊕ RK_matrix[{10-i}]:")
			rk_matrix = create_rk_state_matrix_from_rk(roundkeys[i])	
			print_matrix(rk_matrix)
			state_matrix = matrix_xor_matrix(state_matrix, rk_matrix)
			print("=======>state_matrix<=======")
			print_matrix(state_matrix)
			print(" 2) Mix Column:\nmultiplies 'fixed_matrix' against current state_matrix")
			state_matrix = mix_columns(state_matrix, inv)
			print("=======>state_matrix<=======")
			print_matrix(state_matrix)
			print(" 3) Rows are shifted right-> by offsets of 0,1,2, and 3 =>")
			state_matrix = shift_rows(state_matrix, inv)
			print_matrix(state_matrix)
			print(" 4) Substitution S-box-inv =>")
			state_matrix = matrix_sub_sbox(state_matrix, inv)
			print_matrix(state_matrix)
		
		print(f"\nAdd Roundkey Round 10 ===============================================")		
		print(" state_matrix:")
		print_matrix(state_matrix)
		print(f" ⊕ RK_matrix[0]:")
		rk_matrix = create_rk_state_matrix_from_rk(roundkeys[10])	
		print_matrix(rk_matrix)
		state_matrix = matrix_xor_matrix(state_matrix, rk_matrix)
		print("=======>state_matrix<=======")
		print_matrix(state_matrix)
		print_ciphertext(state_matrix)
		print()
	return state_matrix


def print_plaintext_and_key(plaintext, key, p, k):
	print("--------------------------------------------------------------------")
	print("ASCII Plaintxt ---: '" + ''.join(p) + "'")
	print("ASCII Key --------: '" + ''.join(k) + "'")
	print("HEX   Plaintxt ---: " + ' '.join(plaintext)) 
	print("HEX   Key --------: " + ' '.join(key))
	print("--------------------------------------------------------------------")


def main():
# CHANGE VALUES HERE
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	plaintext = 'Two One Nine Two'#-=-=-=-=-=-=-
	key 	  = 'Thats my Kung Fu'#-=-=-=-=-=-=-
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
	plaintext = plaintext.strip()
	key = key.strip()
	p_hex_str = bytes(plaintext, encoding='utf8').hex()
	k_hex_str = bytes(key, encoding='utf8').hex()
	# split string p into list of 16 pairs of letters
	p_list = [p_hex_str[i:i+2] for i in range(0, 32, 2)]
	k_list = [k_hex_str[i:i+2] for i in range(0, 32, 2)]
	key_hex_list = k_list
	plaintext_hex_list = p_list
	print_plaintext_and_key(plaintext_hex_list, key_hex_list, plaintext, key)

	x = input("""Start process of encrypting then decrypting - Using AES 128-bit cipher.
           
           Press Enter to continue...\n""") 
 
	# state matrix will change throughout encryption and after decryption
	state_matrix = np.array(plaintext_hex_list).reshape(4, 4).T
	# add 10 roundkeys in 10 rounds
	word_list = key_expansion_schedule(key_hex_list)
    # rounkeys = split w into 10 sublists of 4 words
	roundkeys = [''.join(word_list[i]+word_list[i+1]+word_list[i+2]+word_list[i+3]) for i in range(0, len(word_list), 4)]
  	# statematrix = plaintext as 4z4 matrix
	print_keywords(word_list)
	print_round_keys(roundkeys)
	print_rk_state_matrix(roundkeys)

# Encryption
	encrypted_matrix  = deepcopy(add_rounkeys(roundkeys, state_matrix, False))
	state_matrix = deepcopy(encrypted_matrix)
	print("\n====================================================================")
	print("Encrypted Plaintext: ",end='')
	print_matrix_as_list(state_matrix)
	print("====================================================================\n\n")
 
 # Decryption
	decrypted_matrix = deepcopy(add_rounkeys(roundkeys[::-1], state_matrix, True))
	
	print_plaintext_and_key(plaintext_hex_list, key_hex_list, plaintext, key)
	print("-==+1+==- Plaintext: ", end='')
	print_matrix_as_list(decrypted_matrix)
	print("-==+2+==- Encrypted: ", end='')
	print_matrix_as_list(encrypted_matrix)
	print("-==+3+==- Decrypted: ", end='')
	print_matrix_as_list(decrypted_matrix)	
	print()

if __name__ == '__main__':
	main()
#eOF