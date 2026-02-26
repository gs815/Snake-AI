import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque

GRID_SIZE = 10
CELL_SIZE = 30

# --- Reward tuning (modifica questi valori per sperimentare) ---
REWARD_EAT = 5.0            # reward per mangiare il cibo
REWARD_APPROACH = 0.20      # moltiplicatore per riduzione distanza (manhattan)
PENALTY_AWAY = -0.5        # penalità per allontanarsi dal cibo
STEP_PENALTY = -0.03        # penalità per ogni passo (evita looping infinito)
REPEAT_PENALTY = -0.50      # penalità per ripetere posizione della testa recente
MAX_RECENT_HEADS = 8        # quanti head recenti tenere in memoria

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_steps=None):
        super(SnakeEnv, self).__init__()

        self.render_mode = render_mode

        # Azioni: 0=su, 1=giu, 2=sinistra, 3=destra
        self.action_space = spaces.Discrete(4)

        # Osservazione = griglia 10x10, valori in [0,1]
        # 0.0 = vuoto, 0.5 = corpo, 0.75 = testa, 1.0 = cibo
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32
        )

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
            )
            pygame.display.set_caption("Snake RL")

        # max_steps per episodio (evita episodi infiniti)
        self.max_steps = max_steps if max_steps is not None else GRID_SIZE * GRID_SIZE * 6

        # storico recente della testa per rilevare oscillazioni
        self._recent_heads = deque(maxlen=MAX_RECENT_HEADS)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        # direzione iniziale verso il basso (dx,dy)
        self.direction = (0, 1)
        self.food = self._spawn_food()
        self.done = False
        self.steps = 0
        self._recent_heads.clear()
        self._recent_heads.append(self.snake[0])
        self.score = 0  # numero di cibi mangiati

        return self._get_obs(), {}

    def _spawn_food(self):
        # genera posizione valida non occupata dal serpente
        # se la griglia è piena, ritorna None (vittoria)
        free_cells = (GRID_SIZE * GRID_SIZE) - len(self.snake)
        if free_cells <= 0:
            return None

        while True:
            pos = (random.randint(0, GRID_SIZE - 1),
                   random.randint(0, GRID_SIZE - 1))
            if pos not in self.snake:
                return pos

    def _get_obs(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # corpo
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 0.75  # testa
            else:
                grid[y][x] = 0.5   # corpo

        # cibo (se esiste)
        if self.food is not None:
            fx, fy = self.food
            grid[fy][fx] = 1.0

        return grid

    def step(self, action):
        # impedisci inversione immediata (evita 180°)
        opposites = {
            (0, -1): (0, 1),
            (0, 1): (0, -1),
            (-1, 0): (1, 0),
            (1, 0): (-1, 0),
        }

        # mappa azione -> direzione proposta
        proposed = self.direction
        if action == 0:
            proposed = (0, -1)
        elif action == 1:
            proposed = (0, 1)
        elif action == 2:
            proposed = (-1, 0)
        elif action == 3:
            proposed = (1, 0)

        # ignora inversione se è esattamente l'opposto della direzione corrente
        if opposites.get(self.direction) != proposed:
            self.direction = proposed
        # altrimenti mantieni la direzione corrente (no suicidio immediato)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        terminated = False
        truncated = False

        self.steps += 1

        # controllo collisione (muro o sé stesso)
        if (
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake
        ):
            reward = -1.0
            terminated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        # situazione win: se non c'è più posto per il cibo (snake riempie la griglia)
        if self.food is None:
            # gioco vinto
            reward = REWARD_EAT
            terminated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        # distanza manhattan prima e dopo (reward shaping)
        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        proposed_new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        # aggiorna serpente
        self.snake.insert(0, new_head)

        ate = False
        if new_head == self.food:
            # mangiato
            reward += REWARD_EAT
            self.score += 1
            ate = True
            # spawn nuovo cibo
            self.food = self._spawn_food()
        else:
            # non mangiato: rimuovi coda
            self.snake.pop()

        # reward proporzionale all'avvicinamento (positivo se la distanza diminuisce)
        dist_delta = old_distance - proposed_new_distance
        if dist_delta > 0:
            reward += dist_delta * REWARD_APPROACH
        elif dist_delta < 0:
            reward += dist_delta * (-PENALTY_AWAY)  # dist_delta è negativo; applica penalità piccola

        # penalità passo-passiva per scoraggiare loop infiniti o giri inutili
        reward += STEP_PENALTY

        # penalità più forte se si ripete posizione della testa recente (oscillazioni)
        if new_head in list(self._recent_heads):
            # se non ha appena mangiato (se ha mangiato è giustificato riposizionarsi)
            if not ate:
                reward += REPEAT_PENALTY

        # aggiorna storico teste
        self._recent_heads.appendleft(new_head)

        # timeout / truncated se troppi step
        if self.steps >= self.max_steps:
            truncated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        info = {"score": self.score, "steps": self.steps}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        # disegna sfondo
        self.screen.fill((30, 30, 30))

        # disegna celle (serpente e cibo)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                val = 0.0
                # ottieni valore dall'osservazione per coerenza
                if (x, y) in self.snake:
                    if (x, y) == self.snake[0]:
                        val = 0.75
                    else:
                        val = 0.5
                elif (x, y) == self.food:
                    val = 1.0

                if val > 0.0:
                    # colori: testa verde chiaro, corpo verde scuro, cibo rosso
                    if val == 0.75:
                        color = (0, 220, 0)
                    elif val == 0.5:
                        color = (0, 150, 0)
                    else:
                        color = (200, 30, 30)

                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)

        # disegna griglia sottile per chiarezza
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, (50, 50, 50), (0, i * CELL_SIZE), (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))
            pygame.draw.line(self.screen, (50, 50, 50), (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))

        pygame.display.flip()

    def close(self):
        # chiude pygame in modo pulito
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass