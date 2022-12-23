def get_initial_board(rows, columns):
    board = [["." for i in range(columns)] for j in range(rows)]
    return board


def print_board(board):
    print(" " + "_" * (2 * len(board[1]) - 1))
    for i in range(len(board) - 1, -1, -1):
        print("|" + " ".join(board[i]) + "|")
    print(" " + "-" * (2 * len(board[1]) - 1))


def insert_chip(board, column, chip):
    for i in range(len(board)):
        if board[i][column] == ".":
            board[i][column] = chip
            return i
    return -1


def is_win_state(chip, board, row, column):
    # Vertical check
    count = 0
    for i in range(row, -1, -1):
        if board[i][column] == chip:
            count += 1
        elif count > 0:
            count = 0

        if count == 4:
            return True

    # Horizontal check
    count = 0
    for j in range(0, len(board[0])):
        if board[row][j] == chip:
            count += 1
        elif count > 0:
            count = 0

        if count == 4:
            return True

    return False


def is_board_full(board):
    for row in board:
        for unit in row:
            if unit == ".":
                return False

    return True


def main():
    print("Welcome to Find Four!")
    print("---------------------")
    height = -1
    width = -1
    while height not in range(4, 26):
        try:
            height = int(input("Enter height of board (rows): "))
        except:
            print("Error: not a number!")
            continue
        if height > 25:
            print("Error: height can be at most 25!")
        elif height < 4:
            print("Error: height must be at least 4!")
    while width not in range(4, 26):
        try:
            width = int(input("Enter width of board (columns): "))
        except:
            print("Error: not a number!")
            continue
        if width > 25:
            print("Error: width can be at most 25!")
        elif width < 4:
            print("Error: width must be at least 4!")

    board = get_initial_board(height, width)



    print_board(board)
    print("\nPlayer 1: x\nPlayer 2: o")
    # Main game loop:
    end_game = False
    turn = 1
    while not end_game:
        selection = -1
        row = -1
        chip = "x" if turn == 1 else "o"


        while selection not in range(width):
            try:
                selection = int(input(f"Player {turn} - Select a Column: "))
                if selection not in range(width):
                    print("Error: no such column!")
                    continue
                row = insert_chip(board, selection, chip)
                if row == -1:
                    print("Error: column is full!")
                    continue

            except:
                print("Error: not a number! ")





        end_game = is_win_state(chip, board, row, selection)

        if row != -1:
            print_board(board)

            if end_game:
                print("Player", turn, "won the game!")

            turn = 1 if turn == 2 else 2





if __name__ == "__main__":
    main()
