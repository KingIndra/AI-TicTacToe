import pickle, os

filename = 'game_data.pkl'

if os.path.exists(filename):
    with open(filename, 'rb') as f:
        game_data = pickle.load(f)
else:
    game_data = {'no_of_games':0, 'game_space_link':{}, 'game_space':{}}

game_space_link = game_data['game_space_link']
game_space = game_data['game_space']

print('')
print(f'number of games played: {game_data["no_of_games"]}')
print('')
print('GAME_SPACE_LINK: ')
for state in game_space_link:
    print(f'{state} : {game_space_link[state]}') 
print('')
print('GAME_SPACE:')
for state in game_space:
    print(f'{state} : {game_space[state]}')
print('')