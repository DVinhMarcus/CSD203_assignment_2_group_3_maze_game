
import system

def main():
    user_input = input(
        "Choose algorithm (1–4):\n"
        "1. Wilson\n"
        "2. Prim\n"
        "3. Kruskal\n"
        "4. Recursive (DFS)\n> "
    )
    size = int(input("Enter maze size (default 30): "))
    if size >5 and size < 100:
        if user_input == '1':
            system.start_game(ALGORITHM='wilson', MAZE_SIZE=size)
        elif user_input == '2': 
            system.start_game(ALGORITHM='prim', MAZE_SIZE=size)
        elif user_input == '3':
            system.start_game(ALGORITHM='kurskal',MAZE_SIZE=size)
        elif user_input == '4':
            system.start_game(ALGORITHM='dfs', MAZE_SIZE=size    )
        else:
            print("Invalid input. Choose a number from 1 to 4.")

if __name__ == "__main__":
    main()
