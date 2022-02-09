# File mini_rts.py

## Class MiniRTSEnv

## Method __init__()

- Arguments: elf_path, enemyAI, ai_fskip, model_fskip, gpu, num_games, max_ticks

## Method start()

Register action callback
Start minirts

- No Arguments.

## Method step()

Executes action and return observation, done

- Arguments: action

## Method close()

Stops minirts

- No Arguments.

## Method reset()

Restart environment and returns initial observation
TODO: check if self.start() modifies self.observation value

- No Arguments.

## Method restart()

- No Arguments.

## Method _act_callback()

- Arguments: batch

## Method _dummy_train_function()

- Arguments: batch
