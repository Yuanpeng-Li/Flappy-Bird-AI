import pygame
from flappy_bird_gymnasium import FlappyBirdEnv

class HumanFlappyBird(FlappyBirdEnv):
    def __init__(self):
        super(HumanFlappyBird, self).__init__(render_mode="human")
        pygame.init()

    def get_human_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return 1  # Flap
                if event.key == pygame.K_p:
                    self.pause_game()
                if event.key == pygame.K_r:
                    return "restart"
        return 0  # Do nothing

    def pause_game(self):
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        quit()
            pygame.display.update()  # Update the display during pause

env = HumanFlappyBird()
env.close()
while True:
    obs = env.reset()

    while True:
        action = env.get_human_action()
        if action == "restart":
            break
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break
    
    print(info["score"])

