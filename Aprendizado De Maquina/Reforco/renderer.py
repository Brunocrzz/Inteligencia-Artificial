from trainer import Trainer

trainer = Trainer(episodes=5000)

trainer.train()

trainer.evaluate1()

print(trainer.agent.q_table[(1,1,False)])

# {<Action.UP: (-1, 0)>: -14.57154270332934, 
# <Action.DOWN: (1, 0)>: -8.348298380585284, 
# <Action.LEFT: (0, -1)>: -15.706453086995545, 
# <Action.RIGHT: (0, 1)>: 7.694981595263464}

# {<Action.UP: (-1, 0)>: -15.797860545747154, 
# <Action.DOWN: (1, 0)>: -8.39099535217813, 
# <Action.LEFT: (0, -1)>: -15.447919383876643, 
# <Action.RIGHT: (0, 1)>: -6.637680951463007}