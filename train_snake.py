import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv

# -----------------------------
# Ambiente vettorizzato richiesto da SB3
# -----------------------------
env = DummyVecEnv([lambda: SnakeEnv()])

# -----------------------------
# Parametri e path modello
# -----------------------------
MODEL_PATH = "ppo_snake"       # verrà salvato come ppo_snake.zip
total_steps = 3_000_000        # timesteps da eseguire in questa run (sommano se riprendi)

# -----------------------------
# Definiamo la rete della policy più grande (usata solo se creiamo un modello nuovo)
# -----------------------------
policy_kwargs = dict(net_arch=[256, 256])

# -----------------------------
# Carica modello esistente se presente, altrimenti crea uno nuovo
# -----------------------------
model_file_exists = os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")

if model_file_exists:
    try:
        print(f"Found existing model '{MODEL_PATH}'. Loading and continuing training...")
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print("Errore nel caricamento del modello esistente, creerò un nuovo modello. Errore:", e)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003,
            clip_range=0.2,
            n_steps=2048,
            batch_size=64,
        )
else:
    print("Nessun modello trovato. Creo un nuovo modello da zero.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        clip_range=0.2,
        n_steps=2048,
        batch_size=64,
    )

# -----------------------------
# Addestramento (con gestione Ctrl+C per salvare progressi)
# -----------------------------
try:
    print(f"Avvio training per {total_steps:,} timesteps...")
    model.learn(total_timesteps=total_steps)
except KeyboardInterrupt:
    print("\nTraining interrotto dall'utente (KeyboardInterrupt). Salvo modello corrente...")
    model.save(MODEL_PATH)
    print(f"Modello salvato come '{MODEL_PATH}'. Ripeti il training in un secondo momento per continuare.")
    raise

# -----------------------------
# Salvataggio modello addestrato
# -----------------------------
model.save(MODEL_PATH)
print(f"Modello salvato come '{MODEL_PATH}' dopo {total_steps:,} timesteps!")