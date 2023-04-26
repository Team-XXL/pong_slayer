import time

import gymnasium as gym
import numpy as np
import pickle

from collections import namedtuple, defaultdict


class SmartAgent:
    # hyperparameters

    HIDDEN_LAYER = 200  # number of hidden layer neurons
    INPUT_DIMENSION = 6400  # input dimension for the model

    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

    render = False

    save_model = True
    save_interval = 100
    save_file = time.strftime('%Y-%b-%d-%a_%H.%M.%S') + '.csv'

    # resume from previous checkpoint?
    brain_path = 'Pong.pkl'

    transition_ = namedtuple('transition', ('state', 'hidden', 'probability', 'reward'))

    trained_episodes = 2998

    def __init__(self, game_name, resume=False, brain_path=""):
        self.game_name = game_name
        self.brain_path = brain_path
        self.model = self.create_model()
        if resume:
            self.load(self.brain_path)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def discount_rewards(r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        running_add = 0
        discounted_r = np.zeros_like(r)
        for idx in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[idx]
            discounted_r[idx] = running_add
        return discounted_r

    def action(self, obs):
        action_probability, hidden = self.policy_forward(obs)
        action = 2 if action_probability >= 0.5 else 3
        return action

    def load(self, save_path):
        try:
            self.model = pickle.load(open(save_path, 'rb'))
            print("The following model is loaded:", save_path)
        except FileNotFoundError:
            print("The following model could not be found:", save_path)

    def save(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.model, file)

    def create_model(self):
        model = dict()
        model['W1'] = np.random.randn(self.HIDDEN_LAYER, self.INPUT_DIMENSION) / np.sqrt(self.INPUT_DIMENSION)
        model['W2'] = np.random.randn(self.HIDDEN_LAYER) / np.sqrt(self.HIDDEN_LAYER)
        model['TE'] = 0
        return model

    def policy_forward(self, obs):
        """ Return probability of taking action 1 (right), and the hidden state """
        hidden = np.dot(self.model['W1'], obs)
        hidden[hidden < 0] = 0  # ReLU nonlinearity
        log_probability = np.dot(self.model['W2'], hidden)
        probability = self.sigmoid(log_probability)
        return probability, hidden

    def policy_backward(self, episode_observations, episode_hidden, episode_probability):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(episode_hidden.T, episode_probability)
        dh = np.outer(episode_probability, self.model['W2'])
        dh[episode_hidden <= 0] = 0  # backprop relu
        dW1 = np.dot(dh.T, episode_observations)
        return {'W1': dW1, 'W2': dW2}

    def pre_proccessing(self, obs):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0]  # downsample by factor of 2
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        return obs.astype(np.float64).ravel()

    def train(self, nr_games=100):
        env = gym.make(self.game_name)
        running_reward = -21  # self.evaluate(nr_games=5)

        # update buffers that add up gradients over a batch and rmsprop memory
        gradient_buffer = {k: np.zeros_like(v, dtype=np.float64) for k, v in self.model.items()}
        rmsprop_cache = {k: np.zeros_like(v, dtype=np.float64) for k, v in self.model.items()}


        with open(self.save_file, 'a') as file:
                msg = "Episode, Score, Own, Enemy, Mean\n"
                file.write(msg)

        for episode in range(self.model['TE'], nr_games + 1):
            score = defaultdict(int)
            memory = list()
            obs, _ = env.reset()
            done = False

            while not done:
                # Render environment
                if self.render:
                    env.render()

                # Calculate forward policy
                obs = self.pre_proccessing(obs)
                action_probability, hidden = self.policy_forward(obs)
                action = 2 if np.random.uniform() < action_probability else 3

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # grad that encourages the action that was taken to be taken
                # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                probability = (1 if action == 2 else 0) - action_probability
                memory.append(self.transition_(obs, hidden, probability, reward))

                obs = next_obs

                # Pong specific, a point is scored
                if reward:
                    score[reward] += 1
                    msg = "\r\tEpisode {ep:6d}, own {own:2d},  enemy {enemy:2d}, total: {total: 2d}"
                    print(msg.format(ep=episode, own=score[1], enemy=score[-1], total=score[1] - score[-1]), end='')
                self.model['TE'] += 1

            # Convert memory to a stack
            transition = self.transition_(*zip(*memory))
            observations = np.vstack(transition.state)
            hiddens = np.vstack(transition.hidden)
            probabilities = np.hstack(transition.probability)
            rewards = np.hstack(transition.reward)

            # Calculate discounted rewards
            discounter_reward = self.discount_rewards(rewards, self.gamma)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounter_reward -= np.mean(discounter_reward)
            discounter_reward /= np.std(discounter_reward)

            # modulate the gradient with advantage (PG magic happens right here.)
            probabilities *= discounter_reward
            grad = self.policy_backward(observations, hiddens, probabilities)

            # accumulate grad over batch
            for weight in self.model:
                if weight != 'TE':
                    gradient_buffer[weight] += np.array(grad[weight], dtype=np.float64)

            # perform rmsprop parameter update every batch_size episodes
            if episode % self.batch_size == 0:
                for layer, weights in self.model.items():
                    gradient = gradient_buffer[layer]
                    rmsprop_cache[layer] = self.decay_rate * rmsprop_cache[layer] \
                                           + (1 - self.decay_rate) * gradient ** 2

                    self.model[layer] += self.learning_rate * gradient / (np.sqrt(rmsprop_cache[layer]) + 1e-5)
                    gradient_buffer[layer] = np.zeros_like(weights)

            with open(self.save_file, 'a') as file:
                msg = "Episode {ep:6d}, score {s: 2d}, own {o:2d}, enemy {e:2d}, mean {m: 6.2f}\n"
                file.write(msg.format(ep=episode, s=score[1] - score[-1],  o=score[1], e=score[-1], m=running_reward))

            score = score[1] - score[-1]
            running_reward = running_reward * 0.99 + score * 0.01
            print(f"\rEpisode {episode:6d}, score: {score: 4.0f}, running mean: {running_reward: 6.2f}\t\t")

            if self.save_model and episode % self.save_interval == 0:
                self.save(self.brain_path)

        env.close()

    def evaluate(self, nr_games=100):
        """ Evaluate the model results.  """
        env = gym.make(self.game_name, render_mode="human")

        collected_scores = []

        print(f"This model has been trained on {self.model['TE']}")

        for episode in range(1, nr_games + 1):

            obs, _ = env.reset()
            done = False
            score = 0

            while not done:
                # Get action from model
                obs = self.pre_proccessing(obs)
                action = self.action(obs)

                # update everything
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += reward

            print(f"\r\tGame {episode:3d}/{nr_games:3d} score: {score}", end='')

            collected_scores.append(score)

        average = sum(collected_scores) / nr_games
        print(f"\n\nThe model played: {nr_games} games, with an average score of: {average: 5.2f}")
        return average