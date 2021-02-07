"""
For human play
"""
import env
actionSpace = ('hit', 'stick')


def get_input_action():
    action = input('What\'s your move? (hit or stick)')
    while action not in actionSpace:
        action = input('Either say hit or stick please!')
    return action


s, _ = env.start()
print('Let\'s start!')
while s != 'terminal':
    print(s)
    a = get_input_action()
    s, r = env.step(s, a)

if r == -1:
    print("You lost")
elif r == 0:
    print("Draw")
else:
    print("You won")